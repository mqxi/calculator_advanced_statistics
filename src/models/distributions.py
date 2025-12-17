import numpy as np
import sympy as sp
from scipy.stats import binom, geom, nbinom, norm, poisson, weibull_min

from .base import CalculationResult, CalculationStep, DistributionModel


class NormalDistribution(DistributionModel):
    def __init__(self):
        super().__init__("Normalverteilung", "continuous")

        # SymPy Symbole definieren
        self.x, self.mu, self.sigma = sp.symbols(r"x \mu \sigma")
        # Symbolische Formel aufbauen
        # Wir nutzen sp.exp und sp.sqrt für die Darstellung
        term1 = 1 / (self.sigma * sp.sqrt(2 * sp.pi))
        term2 = sp.exp(-sp.Rational(1, 2) * ((self.x - self.mu) / self.sigma) ** 2)
        self.expr = term1 * term2

    def get_variables(self):
        # Das GUI wird diese Keys nutzen, um Input-Felder zu erstellen
        return [
            {"name": "x", "label": "Wert (x)", "default": 0.0},
            {"name": "mu", "label": "Mittelwert (μ)", "default": 0.0},
            {"name": "sigma", "label": "Standardabw. (σ)", "default": 1.0},
        ]

    def get_formula_latex(self) -> str:
        return r"f(x) = " + sp.latex(self.expr)

    def calculate(self, x: float, mu: float, sigma: float) -> CalculationResult:
        """
        Berechnet P(X=x) bzw. die Dichte f(x) und liefert den Rechenweg.
        """
        if sigma <= 0:
            raise ValueError("Standardabweichung sigma muss > 0 sein.")

        steps = []

        # 1. Allgemeine Formel
        steps.append(
            CalculationStep(
                description="Allgemeine Dichtefunktion der Normalverteilung:",
                latex=self.get_formula_latex(),
            )
        )

        # 2. Einsetzen (Substitution)
        # Wir bauen den String manuell für optimale Lesbarkeit, da SymPy oft zu früh vereinfacht
        sub_latex = (
            r"f(" + f"{x}" + r") = \frac{1}{" + f"{sigma}" + r"\sqrt{2\pi}} "
            r"\cdot e^{-\frac{1}{2}\left(\frac{"
            + f"{x} - {mu}"
            + r"}{"
            + f"{sigma}"
            + r"}\right)^2}"
        )
        steps.append(
            CalculationStep(
                description=f"Einsetzen der Werte (x={x}, μ={mu}, σ={sigma}):",
                latex=sub_latex,
            )
        )

        # 3. Z-Score Berechnung (Zwischenschritt im Exponenten)
        z_score = (x - mu) / sigma
        exponent = -0.5 * (z_score**2)
        coeff = 1 / (sigma * np.sqrt(2 * np.pi))

        inter_latex = (
            r"= \frac{1}{" + f"{sigma * np.sqrt(2 * np.pi):.4f}" + r"} "
            r"\cdot e^{" + f"{exponent:.4f}" + r"}"
        )
        steps.append(
            CalculationStep(
                description="Vereinfachung des Vorfaktors und des Exponenten:",
                latex=inter_latex,
            )
        )

        # 4. Ergebnis
        result_val = norm.pdf(x, loc=mu, scale=sigma)
        steps.append(
            CalculationStep(
                description="Endergebnis (Wahrscheinlichkeitsdichte):",
                latex=r"= " + f"{result_val:.6f}",
            )
        )

        # Plot Daten generieren
        # Bereich: +/- 4 Standardabweichungen
        x_axis = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
        y_axis = norm.pdf(x_axis, mu, sigma)

        plot_data = {
            "x": x_axis,
            "y": y_axis,
            "highlight_x": x,
            "highlight_y": result_val,
            "xlabel": "x",
            "ylabel": "Dichte f(x)",
        }

        return CalculationResult(result=result_val, steps=steps, plot_data=plot_data)


class BinomialDistribution(DistributionModel):
    def __init__(self):
        super().__init__("Binomialverteilung", "discrete")
        self.n, self.k, self.p = sp.symbols("n k p")
        # Binomialkoeffizient symbolisch
        self.expr = (
            sp.binomial(self.n, self.k)
            * (self.p**self.k)
            * ((1 - self.p) ** (self.n - self.k))
        )

    def get_variables(self):
        return [
            {"name": "n", "label": "Versuche (n)", "default": 10},
            {"name": "p", "label": "Erfolgswahrsch. (p)", "default": 0.5},
            {"name": "k", "label": "Treffer (k)", "default": 5},
        ]

    def get_formula_latex(self) -> str:
        return r"P(X=k) = " + sp.latex(self.expr)

    def calculate(self, n: int, p: float, k: int) -> CalculationResult:
        # Typkonvertierung sicherstellen
        n = int(n)
        k = int(k)

        if not (0 <= p <= 1):
            raise ValueError("Wahrscheinlichkeit p muss zwischen 0 und 1 liegen.")
        if k < 0 or k > n:
            raise ValueError("k muss zwischen 0 und n liegen.")

        steps = []

        # 1. Formel
        steps.append(
            CalculationStep(
                description="Wahrscheinlichkeitsfunktion (PMF):",
                latex=self.get_formula_latex(),
            )
        )

        # 2. Substitution
        binom_coeff_val = sp.binomial(n, k)  # Wert berechnen für Anzeige
        sub_latex = (
            r"P(X=" + f"{k}" + r") = \binom{" + f"{n}" + r"}{" + f"{k}" + r"} "
            r"\cdot "
            + f"{p}"
            + r"^{"
            + f"{k}"
            + r"} \cdot (1 - "
            + f"{p}"
            + r")^{"
            + f"{n}-{k}"
            + r"}"
        )
        steps.append(
            CalculationStep(
                description=f"Einsetzen (n={n}, p={p}, k={k}):", latex=sub_latex
            )
        )

        # 3. Zwischenschritt
        p_term = p**k
        q_term = (1 - p) ** (n - k)
        inter_latex = (
            r"= "
            + f"{binom_coeff_val}"
            + r" \cdot "
            + f"{p_term:.4f}"
            + r" \cdot "
            + f"{q_term:.4f}"
        )
        steps.append(
            CalculationStep(
                description="Berechnung der Komponenten (Binomialkoeffizient * p^k * q^(n-k)):",
                latex=inter_latex,
            )
        )

        # 4. Ergebnis
        result_val = binom.pmf(k, n, p)
        steps.append(
            CalculationStep(
                description="Endergebnis:", latex=r"= " + f"{result_val:.6f}"
            )
        )

        # Plot Daten (Diskret -> wir brauchen Integer Schritte für x)
        x_axis = np.arange(0, n + 1)
        y_axis = binom.pmf(x_axis, n, p)

        plot_data = {
            "x": x_axis,
            "y": y_axis,
            "highlight_x": k,
            "highlight_y": result_val,
            "xlabel": "Anzahl Treffer (k)",
            "ylabel": "Wahrscheinlichkeit P(X=k)",
        }

        return CalculationResult(result=result_val, steps=steps, plot_data=plot_data)


class WeibullDistribution(DistributionModel):
    def __init__(self):
        super().__init__("Weibull Verteilung", "continuous")

        # Symbole: k (Form), lambda (Skala), x
        self.x, self.k, self.lam = sp.symbols(r"x k \lambda")

        # Formel: f(x) = (k/lambda) * (x/lambda)^(k-1) * e^(-(x/lambda)^k)
        # Hinweis: Für x >= 0
        term1 = self.k / self.lam
        term2 = (self.x / self.lam) ** (self.k - 1)
        term3 = sp.exp(-((self.x / self.lam) ** self.k))

        self.expr = term1 * term2 * term3

    def get_variables(self):
        return [
            {"name": "x", "label": "Wert (x)", "default": 1.5, "min": 0.0},
            {"name": "k", "label": "Formparameter (k)", "default": 1.5, "min": 0.1},
            {"name": "lam", "label": "Skalenparameter (λ)", "default": 1.0, "min": 0.1},
        ]

    def get_formula_latex(self) -> str:
        return r"f(x) = \frac{k}{\lambda} \left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}"

    def calculate(self, x: float, k: float, lam: float) -> CalculationResult:
        # Validierung
        if x < 0:
            return CalculationResult(
                0,
                [
                    CalculationStep(
                        "Info", "Die Weibull-Verteilung ist für x < 0 definiert als 0."
                    )
                ],
            )
        if k <= 0 or lam <= 0:
            raise ValueError("Parameter k und λ müssen positiv sein.")

        steps = []

        # 1. Formel anzeigen
        steps.append(
            CalculationStep(
                description="Dichtefunktion der Weibull-Verteilung:",
                latex=self.get_formula_latex() + r", \quad x \geq 0",
            )
        )

        # 2. Einsetzen
        sub_latex = (
            r"f(" + f"{x}" + r") = \frac{" + f"{k}" + r"}{" + f"{lam}" + r"} "
            r"\left(\frac{"
            + f"{x}"
            + r"}{"
            + f"{lam}"
            + r"}\right)^{"
            + f"{k}-1"
            + r"} "
            r"e^{-\left(\frac{"
            + f"{x}"
            + r"}{"
            + f"{lam}"
            + r"}\right)^{"
            + f"{k}"
            + r"}}"
        )
        steps.append(
            CalculationStep(
                description=f"Einsetzen der Werte (x={x}, k={k}, λ={lam}):",
                latex=sub_latex,
            )
        )

        # 3. Zwischenschritte berechnen
        ratio = x / lam
        exponent_term = ratio**k
        pre_factor = (k / lam) * (ratio ** (k - 1))

        inter_latex = (
            r"= "
            + f"{k / lam:.4f}"
            + r" \cdot ("
            + f"{ratio:.4f}"
            + r")^{"
            + f"{k - 1:.4f}"
            + r"} "
            r"\cdot e^{-" + f"{exponent_term:.4f}" + r"}"
        )
        steps.append(
            CalculationStep(description="Vereinfachung der Terme:", latex=inter_latex)
        )

        # 4. Endergebnis
        # In Scipy ist 'c' der Formparameter (k) und 'scale' ist lambda
        result_val = weibull_min.pdf(x, c=k, scale=lam)

        steps.append(
            CalculationStep(
                description="Endergebnis:", latex=r"= " + f"{result_val:.6f}"
            )
        )

        # Plot Daten generieren
        # Weibull startet bei 0. Wir plotten bis mean + 4*std oder mindestens bis x*1.5
        mean = lam * sp.gamma(1 + 1 / k)
        # Wir nutzen numpy für Gamma, da sympy Gamma float zurückgibt
        mean_val = float(mean)
        limit = max(x * 1.5, mean_val * 3)

        x_axis = np.linspace(0, limit, 400)
        y_axis = weibull_min.pdf(x_axis, c=k, scale=lam)

        plot_data = {
            "x": x_axis,
            "y": y_axis,
            "highlight_x": x,
            "highlight_y": result_val,
            "xlabel": "x",
            "ylabel": "Dichte f(x)",
            "title": f"Weibull-Verteilung (k={k}, λ={lam})",
        }

        return CalculationResult(result=result_val, steps=steps, plot_data=plot_data)


class PoissonDistribution(DistributionModel):
    def __init__(self):
        super().__init__("Poisson Verteilung", "discrete")
        self.lam, self.k = sp.symbols(r"\lambda k")
        # P(X=k) = (lambda^k * e^-lambda) / k!
        self.expr = (self.lam**self.k * sp.exp(-self.lam)) / sp.factorial(self.k)

    def get_variables(self):
        return [
            {"name": "lam", "label": "Lambda (λ)", "default": 3.0, "min": 0.1},
            {"name": "k", "label": "Ereignisse (k)", "default": 2, "step": 1, "min": 0},
        ]

    def get_formula_latex(self) -> str:
        return r"P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}"

    def calculate(self, lam: float, k: int) -> CalculationResult:
        k = int(k)
        if lam <= 0:
            raise ValueError("Lambda muss positiv sein.")
        if k < 0:
            raise ValueError("k muss >= 0 sein.")

        steps = []

        # 1. Formel
        steps.append(
            CalculationStep("Formel der Poisson-Verteilung:", self.get_formula_latex())
        )

        # 2. Einsetzen
        sub_latex = (
            r"P(X="
            + f"{k}"
            + r") = \frac{"
            + f"{lam}^{k}"
            + r" \cdot e^{-"
            + f"{lam}"
            + r"}}{"
            + f"{k}!"
            + r"}"
        )
        steps.append(CalculationStep(f"Einsetzen (λ={lam}, k={k}):", sub_latex))

        # 3. Ergebnis
        result_val = poisson.pmf(k, mu=lam)
        steps.append(CalculationStep("Ergebnis:", r"= " + f"{result_val:.6f}"))

        # Plot (Diskrete Balken)
        # Wir plotten bis die Wahrscheinlichkeit sehr klein wird (Poisson Tail)
        limit = int(poisson.ppf(0.999, lam)) + 2
        x_axis = np.arange(0, limit)
        y_axis = poisson.pmf(x_axis, mu=lam)

        plot_data = {
            "x": x_axis,
            "y": y_axis,
            "highlight_x": k,
            "highlight_y": result_val,
            "xlabel": "Anzahl Ereignisse (k)",
            "ylabel": "P(X=k)",
            "title": f"Poisson-Verteilung (λ={lam})",
        }

        return CalculationResult(result_val, steps, plot_data)


class GeometricDistribution(DistributionModel):
    def __init__(self):
        super().__init__("Geometrische Verteilung", "discrete")
        self.p, self.k = sp.symbols("p k")
        # P(X=k) = (1-p)^(k-1) * p  (Versuche bis zum ersten Erfolg)
        self.expr = (1 - self.p) ** (self.k - 1) * self.p

    def get_variables(self):
        return [
            {
                "name": "p",
                "label": "Erfolgswahrsch. (p)",
                "default": 0.5,
                "min": 0.01,
                "max": 1.0,
            },
            {
                "name": "k",
                "label": "Versuch Nr. (k)",
                "default": 1,
                "step": 1,
                "min": 1,
            },
        ]

    def get_formula_latex(self) -> str:
        return r"P(X=k) = (1-p)^{k-1} \cdot p"

    def calculate(self, p: float, k: int) -> CalculationResult:
        k = int(k)
        if not (0 < p <= 1):
            raise ValueError("p muss zwischen 0 und 1 liegen.")
        if k < 1:
            raise ValueError("k muss >= 1 sein (erster Versuch).")

        steps = []
        steps.append(
            CalculationStep(
                "Formel (Anzahl Versuche bis Erfolg):", self.get_formula_latex()
            )
        )

        sub_latex = (
            r"P(X="
            + f"{k}"
            + r") = (1 - "
            + f"{p}"
            + r")^{"
            + f"{k}-1"
            + r"} \cdot "
            + f"{p}"
        )
        steps.append(CalculationStep(f"Einsetzen (p={p}, k={k}):", sub_latex))

        result_val = geom.pmf(k, p)
        steps.append(CalculationStep("Ergebnis:", r"= " + f"{result_val:.6f}"))

        # Plot
        limit = int(geom.ppf(0.999, p)) + 2
        x_axis = np.arange(1, limit)  # Geometrisch startet bei 1
        y_axis = geom.pmf(x_axis, p)

        plot_data = {
            "x": x_axis,
            "y": y_axis,
            "highlight_x": k,
            "highlight_y": result_val,
            "xlabel": "Anzahl Versuche (k)",
            "ylabel": "P(X=k)",
            "title": f"Geometrische Verteilung (p={p})",
        }
        return CalculationResult(result_val, steps, plot_data)


class NegativeBinomialDistribution(DistributionModel):
    def __init__(self):
        super().__init__("Negative Binomialverteilung", "discrete")
        # Definition: Anzahl Misserfolge (k) bis r Erfolge eintreten.
        # Achtung: Workbook sagt "Anzahl Meteoriten" bei gegebenen "Anzahl Erfolgen k_param".
        # Wir nutzen die Standardform: k = Anzahl Misserfolge (x-Achse), n = Anzahl Erfolge (fixer Parameter)
        self.n, self.k, self.p = sp.symbols("n k p")
        # P(X=k) = binom(k+n-1, n-1) * p^n * (1-p)^k
        self.expr = (
            sp.binomial(self.k + self.n - 1, self.n - 1)
            * self.p**self.n
            * (1 - self.p) ** self.k
        )

    def get_variables(self):
        return [
            {
                "name": "n_success",
                "label": "Benötigte Erfolge (r)",
                "default": 5,
                "step": 1,
                "min": 1,
            },
            {
                "name": "p",
                "label": "Erfolgswahrsch. (p)",
                "default": 0.5,
                "min": 0.01,
                "max": 1.0,
            },
            {
                "name": "k_failures",
                "label": "Anzahl Misserfolge (k)",
                "default": 3,
                "step": 1,
                "min": 0,
            },
        ]

    def get_formula_latex(self) -> str:
        # Wir nutzen 'r' für Erfolge im LaTeX, damit es klarer ist, da 'n' oft Versuche sind
        return r"P(X=k) = \binom{k+r-1}{r-1} p^r (1-p)^k"

    def calculate(self, n_success: int, p: float, k_failures: int) -> CalculationResult:
        r = int(n_success)  # Das Workbook nennt dies oft k, wir nennen es r (Standard)
        k = int(k_failures)  # Das ist unsere Zufallsvariable (Meteoriten?)

        if not (0 < p <= 1):
            raise ValueError("p muss zwischen 0 und 1 liegen.")
        if r < 1:
            raise ValueError("Anzahl Erfolge muss >= 1 sein.")
        if k < 0:
            raise ValueError("Misserfolge k muss >= 0 sein.")

        steps = []
        steps.append(
            CalculationStep(
                "Formel (Wahrscheinlichkeit für k Misserfolge bevor r Erfolge):",
                self.get_formula_latex(),
            )
        )

        # Einsetzen
        binom_part = sp.binomial(k + r - 1, r - 1)
        sub_latex = (
            r"P(X=" + f"{k}" + r") = \binom{" + f"{k}+{r}-1" + r"}{" + f"{r}-1" + r"} "
            r"\cdot " + f"{p}^{r}" + r" \cdot (1 - " + f"{p}" + r")^{" + f"{k}" + r"}"
        )
        steps.append(CalculationStep(f"Einsetzen (r={r}, p={p}, k={k}):", sub_latex))

        # Zwischenschritt
        inter_latex = (
            r"= "
            + f"{binom_part}"
            + r" \cdot "
            + f"{p**r:.4f}"
            + r" \cdot "
            + f"{(1 - p) ** k:.4f}"
        )
        steps.append(CalculationStep("Komponenten berechnen:", inter_latex))

        # Ergebnis (Scipy nbinom nutzt n=Erfolge, p=Erfolgswahrsch)
        result_val = nbinom.pmf(k, r, p)
        steps.append(CalculationStep("Ergebnis:", r"= " + f"{result_val:.6f}"))

        # Plot
        limit = int(nbinom.ppf(0.999, r, p)) + 2
        x_axis = np.arange(0, limit)
        y_axis = nbinom.pmf(x_axis, r, p)

        plot_data = {
            "x": x_axis,
            "y": y_axis,
            "highlight_x": k,
            "highlight_y": result_val,
            "xlabel": "Anzahl Misserfolge (k)",
            "ylabel": "P(X=k)",
            "title": f"Neg. Binomial (r={r}, p={p})",
        }
        return CalculationResult(result_val, steps, plot_data)


class BernoulliDistribution(DistributionModel):
    def __init__(self):
        super().__init__("Bernoulli Verteilung", "discrete")
        self.p, self.k = sp.symbols("p k")
        # P(X=1) = p, P(X=0) = 1-p
        # Kompakte Formel: p^k * (1-p)^(1-k) für k in {0,1}
        self.expr = self.p**self.k * (1 - self.p) ** (1 - self.k)

    def get_variables(self):
        return [
            {
                "name": "p",
                "label": "Erfolgswahrsch. (p)",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
            },
            {
                "name": "k",
                "label": "Ergebnis (1=Ja, 0=Nein)",
                "default": 1,
                "step": 1,
                "min": 0,
                "max": 1,
            },
        ]

    def get_formula_latex(self) -> str:
        return r"P(X=k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}"

    def calculate(self, p: float, k: int) -> CalculationResult:
        steps = []
        val = p if k == 1 else 1 - p

        steps.append(
            CalculationStep(
                "Wahrscheinlichkeit:",
                rf"P(X={k}) = {val:.4f} \quad (\text{{{'Dafür' if k == 1 else 'Dagegen'}}})",
            )
        )

        # Erwartungswert E[X] = p
        steps.append(CalculationStep("Erwartungswert E[X]:", rf"E[X] = p = {p:.4f}"))

        # Varianz Var(X) = p(1-p)
        var = p * (1 - p)
        steps.append(
            CalculationStep("Varianz Var(X):", rf"\text{{Var}}(X) = p(1-p) = {var:.4f}")
        )

        # Plot (Balken)
        plot_data = {
            "x": [0, 1],
            "y": [1 - p, p],
            "highlight_x": k,
            "highlight_y": val,
            "xlabel": "Ergebnis (0=Nein, 1=Ja)",
            "ylabel": "Wahrscheinlichkeit",
            "title": f"Bernoulli-Verteilung (p={p})",
        }
        return CalculationResult(val, steps, plot_data)


class ExponentialDistribution(DistributionModel):
    def __init__(self):
        super().__init__("Exponentialverteilung", "continuous")
        self.x, self.lam = sp.symbols(r"x \lambda")
        # f(x) = lambda * exp(-lambda * x)
        self.expr = self.lam * sp.exp(-self.lam * self.x)

    def get_variables(self):
        return [
            {"name": "x", "label": "Wert (x)", "default": 1.0, "min": 0.0},
            {"name": "lam", "label": "Rate (λ)", "default": 1.0, "min": 0.0001},
        ]

    def get_formula_latex(self) -> str:
        return r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0"

    def calculate(self, x: float, lam: float) -> CalculationResult:
        if x < 0:
            return CalculationResult(
                0, [CalculationStep("Info", "Für x < 0 ist f(x) = 0.")]
            )
        if lam <= 0:
            raise ValueError("Rate λ muss positiv sein.")

        steps = []
        steps.append(
            CalculationStep(
                "Dichtefunktion:",
                self.get_formula_latex(),
            )
        )

        sub_latex = (
            r"f(" + f"{x}" + r") = " + f"{lam}" + r" \cdot e^{-" + f"{lam}" + r" \cdot " + f"{x}" + r"}"
        )
        steps.append(CalculationStep(f"Einsetzen (x={x}, λ={lam}):", sub_latex))

        exponent = -lam * x
        inter_latex = r"= " + f"{lam}" + r" \cdot e^{" + f"{exponent:.4f}" + r"}"
        steps.append(CalculationStep("Exponent berechnen:", inter_latex))

        from scipy.stats import expon
        # scipy uses scale = 1/lambda
        result_val = expon.pdf(x, scale=1.0/lam)
        steps.append(CalculationStep("Ergebnis:", r"= " + f"{result_val:.6f}"))

        # Plot
        # Plot up to where PDF drops significantly, e.g., 4 * mean
        mean = 1.0 / lam
        limit = max(x * 1.5, mean * 4)
        x_axis = np.linspace(0, limit, 400)
        y_axis = expon.pdf(x_axis, scale=1.0/lam)

        plot_data = {
            "x": x_axis,
            "y": y_axis,
            "highlight_x": x,
            "highlight_y": result_val,
            "xlabel": "x",
            "ylabel": "Dichte f(x)",
            "title": f"Exponentialverteilung (λ={lam})",
        }

        return CalculationResult(result_val, steps, plot_data)
