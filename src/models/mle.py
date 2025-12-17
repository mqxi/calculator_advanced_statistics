import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar

from .base import CalculationResult, CalculationStep, StatisticalModel


class MaximumLikelihoodEstimation(StatisticalModel):
    def __init__(self):
        super().__init__("Maximum Likelihood (MLE)")
        self.x_sym = sp.Symbol("x")

    def get_variables(self):
        return [
            # Daten aus dem Workbook (z.B. xi_10)
            {
                "name": "data_str",
                "label": "Datenstichprobe (kommagetrennt)",
                "default": "1.5, 2.3, 1.8, 2.9, 3.1",
                "type": "text",
            },
            # Die Formel für T (muss der User herleiten, z.B. Gamma für Summe von Exponentials)
            {
                "name": "pdf_str",
                "label": "Dichtefunktion f(x; theta)",
                "default": "theta**2 * x * exp(-theta * x)",
                "type": "text",
            },
            {
                "name": "param_name",
                "label": "Parameter Name",
                "default": "theta",
                "type": "text",
            },
            {"name": "search_min", "label": "Suche Min", "default": 0.1},
            {"name": "search_max", "label": "Suche Max", "default": 5.0},
        ]

    def get_formula_latex(self) -> str:
        return r"\hat{\theta}_{MLE} = \operatorname{argmax}_\theta \mathcal{L}(\theta)"

    def calculate(
        self,
        data_str: str,
        pdf_str: str,
        param_name: str,
        search_min: float,
        search_max: float,
    ) -> CalculationResult:
        steps = []

        # 1. Daten parsen
        try:
            data = np.array([float(x.strip()) for x in data_str.split(",")])
            n = len(data)
        except:
            return CalculationResult(0, [CalculationStep("Fehler", "Ungültige Daten.")])

        # 2. Funktion parsen
        try:
            # Wir brauchen x und den Parameter (z.B. theta) als Symbole
            theta_sym = sp.Symbol(param_name)
            # local_dict für sicheres Parsen
            local_dict = {
                "x": self.x_sym,
                param_name: theta_sym,
                "exp": sp.exp,
                "log": sp.log,
                "sin": sp.sin,
                "sqrt": sp.sqrt,
                "pi": sp.pi,
            }
            pdf_expr = sp.sympify(pdf_str, locals=local_dict)
        except Exception as e:
            return CalculationResult(
                0, [CalculationStep("Fehler", f"Formel Fehler: {e}")]
            )

        # LaTeX der PDF anzeigen
        steps.append(
            CalculationStep(
                description="Modell-Dichtefunktion (PDF) für eine Beobachtung:",
                latex=r"f(x; " + param_name + r") = " + sp.latex(pdf_expr),
            )
        )

        # 3. Likelihood Funktion aufstellen (Symbolisch)
        # L(theta) = Prod f(xi; theta)
        steps.append(
            CalculationStep(
                description="Likelihood-Funktion (Produkt der Wahrscheinlichkeiten):",
                latex=r"\mathcal{L}("
                + param_name
                + r") = \prod_{i=1}^{n} f(x_i; "
                + param_name
                + r")",
            )
        )

        # 4. Log-Likelihood Transformation (Das ist die Antwort für Aufgabe 3!)
        steps.append(
            CalculationStep(
                description="Transformation zur Log-Likelihood (wendet Logarithmus an, Maximum bleibt gleich):",
                latex=r"\ell("
                + param_name
                + r") = \ln(\mathcal{L}) = \sum_{i=1}^{n} \ln(f(x_i; "
                + param_name
                + r"))",
            )
        )

        # 5. Numerische Optimierung
        # Wir wandeln den Sympy-Ausdruck in eine Python-Funktion f(x, theta)
        func_num = sp.lambdify((self.x_sym, theta_sym), pdf_expr, "numpy")

        # Zielfunktion: Negative Log-Likelihood (weil Optimizer minimieren wollen)
        def neg_log_likelihood(theta_val):
            if theta_val <= 0:
                return 1e9  # Schutz vor log(0)
            # pdf Werte berechnen
            vals = func_num(data, theta_val)
            # Schutz vor log(0) oder negativen Werten in PDF
            vals = np.maximum(vals, 1e-15)
            return -np.sum(np.log(vals))

        # Suche das Minimum der negativen LL (= Maximum der LL)
        res = minimize_scalar(
            neg_log_likelihood, bounds=(search_min, search_max), method="bounded"
        )

        if res.success:
            best_theta = res.x
            max_log_lik = -res.fun

            steps.append(
                CalculationStep(
                    description="Numerische Maximierung (MLE):",
                    latex=r"\hat{"
                    + param_name
                    + r"}_{MLE} \approx "
                    + f"{best_theta:.5f}",
                )
            )

            steps.append(
                CalculationStep(
                    description="Maximaler Log-Likelihood Wert:",
                    latex=r"\ell_{max} = " + f"{max_log_lik:.4f}",
                )
            )
        else:
            return CalculationResult(
                0, [CalculationStep("Fehler", "Optimierung fehlgeschlagen.")]
            )

        # --- PLOTTING (Transformation visualisieren) ---
        # Wir plotten Likelihood UND Log-Likelihood, um zu zeigen, dass die Peaks alignen.
        theta_range = np.linspace(search_min, search_max, 200)

        # Vektorisierte Berechnung für Plot
        ll_values = []
        lik_values = []  # Achtung: Likelihood wird extrem klein (Underflow Gefahr), wir loggen es meist

        for t_val in theta_range:
            val = -neg_log_likelihood(t_val)
            ll_values.append(val)
            # Für die Likelihood-Kurve "faken" wir es etwas für die Visualisierung,
            # da exp(-1000) 0 ist. Wir plotten exp(val - max_val), also relative Likelihood.
            lik_values.append(np.exp(val - max_log_lik))

        # Wir normalisieren beide Kurven auf [0, 1] für den Vergleich im Plot
        ll_norm = (ll_values - np.min(ll_values)) / (
            np.max(ll_values) - np.min(ll_values)
        )
        lik_norm = np.array(lik_values)  # Ist schon max 1 durch den Trick oben

        plot_data = {
            "x": theta_range,
            "y_prior": lik_norm,  # Wir missbrauchen y_prior für Likelihood (gestrichelt)
            "y_post": ll_norm,  # Wir missbrauchen y_post für Log-Likelihood (durchgezogen)
            "highlight_x": best_theta,
            "xlabel": f"Parameter {param_name}",
            "ylabel": "Normalisierter Wert (Skaliert)",
            "title": "Vergleich: Likelihood vs. Log-Likelihood (Peaks stimmen überein)",
            "legend_labels": ["Likelihood L (relativ)", "Log-Likelihood l (normiert)"],
        }

        # Hinweis: PlotCanvas muss angepasst werden, um legend_labels zu unterstützen,
        # oder wir nutzen die Standard-Labels von _plot_bayes, da wir dessen Datenstruktur nutzen.
        # Einfacher Hack: Wir nutzen _plot_bayes Struktur, da die Linienfarben passen.

        return CalculationResult(
            best_theta, steps, plot_data, meta_data={"type": "bayesian_plot"}
        )
