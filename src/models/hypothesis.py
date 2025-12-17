import numpy as np
import sympy as sp
from scipy.stats import chi2, f, t

from .base import CalculationResult, CalculationStep, StatisticalModel


class OneSampleTTest(StatisticalModel):
    def __init__(self):
        super().__init__("Einstichproben t-Test")

        # Symbole definieren
        self.mu, self.mu0 = sp.symbols(r"\mu \mu_0")
        self.x_bar = sp.symbols(r"\bar{x}")
        self.s = sp.symbols("s")
        self.n = sp.symbols("n")
        self.t_stat = sp.symbols("t")

        # Formel für t-Statistik
        # t = (mean - mu0) / (s / sqrt(n))
        self.expr = (self.x_bar - self.mu0) / (self.s / sp.sqrt(self.n))

    def get_variables(self):
        return [
            {"name": "x_bar", "label": "Stichprobenmittel (x̄)", "default": 105.0},
            {"name": "s", "label": "Standardabw. (s)", "default": 15.0},
            {"name": "n", "label": "Stichprobengröße (n)", "default": 30, "step": 1},
            {"name": "mu_0", "label": "Hypothese (μ₀)", "default": 100.0},
            {"name": "alpha", "label": "Signifikanzniveau (α)", "default": 0.05},
            {
                "name": "alternative",
                "label": "Alternative H₁",
                "default": "two-sided",
                "type": "dropdown",
                "options": [
                    ("Ungleich (≠)", "two-sided"),
                    ("Größer (>)", "greater"),
                    ("Kleiner (<)", "less"),
                ],
            },
        ]

    def get_formula_latex(self) -> str:
        return r"t = " + sp.latex(self.expr)

    def calculate(
        self,
        x_bar: float,
        s: float,
        n: int,
        mu_0: float,
        alpha: float,
        alternative: str,
    ) -> CalculationResult:
        n = int(n)
        steps = []

        # --- Schritt 1: Hypothesen aufstellen ---
        if alternative == "two-sided":
            h0 = r"H_0: \mu = " + str(mu_0)
            h1 = r"H_1: \mu \neq " + str(mu_0)
            sign_str = r"\neq"
        elif alternative == "greater":
            h0 = r"H_0: \mu \leq " + str(mu_0)
            h1 = r"H_1: \mu > " + str(mu_0)
            sign_str = ">"
        else:  # less
            h0 = r"H_0: \mu \geq " + str(mu_0)
            h1 = r"H_1: \mu < " + str(mu_0)
            sign_str = "<"

        steps.append(
            CalculationStep(
                description="Aufstellen der Hypothesen:",
                latex=f"{h0} \\quad \\text{{vs.}} \\quad {h1}",
            )
        )

        # --- Schritt 2: Standardfehler (SE) ---
        se = s / np.sqrt(n)
        se_latex = (
            r"SE = \frac{s}{\sqrt{n}} = \frac{"
            + f"{s}"
            + r"}{\sqrt{"
            + f"{n}"
            + r"}} = "
            + f"{se:.4f}"
        )

        steps.append(
            CalculationStep(
                description="Berechnung des Standardfehlers (Standard Error):",
                latex=se_latex,
            )
        )

        # --- Schritt 3: t-Statistik Berechnung ---
        t_val = (x_bar - mu_0) / se

        # Formel substitution
        sub_latex = (
            r"t = \frac{" + f"{x_bar} - {mu_0}" + r"}{" + f"{se:.4f}" + r"} = "
            r"\frac{" + f"{x_bar - mu_0:.4f}" + r"}{" + f"{se:.4f}" + r"}"
        )

        steps.append(
            CalculationStep(
                description="Berechnung der Teststatistik (t-Wert):",
                latex=sub_latex + r" = " + f"{t_val:.4f}",
            )
        )

        # --- Schritt 4: Freiheitsgrade und p-Wert ---
        df = n - 1

        if alternative == "two-sided":
            # P-Wert ist 2 * Fläche am Rand
            p_val = 2 * (1 - t.cdf(abs(t_val), df))
            p_calc_latex = r"p = 2 \cdot (1 - P(T \leq |" + f"{t_val:.4f}" + r"|))"
        elif alternative == "greater":
            p_val = 1 - t.cdf(t_val, df)
            p_calc_latex = r"p = P(T > " + f"{t_val:.4f}" + r")"
        else:  # less
            p_val = t.cdf(t_val, df)
            p_calc_latex = r"p = P(T < " + f"{t_val:.4f}" + r")"

        steps.append(
            CalculationStep(
                description=f"Berechnung des p-Werts (mit df={df}):",
                latex=p_calc_latex + r" = " + f"{p_val:.6f}",
            )
        )

        # --- Schritt 5: Entscheidung ---
        decision = (
            "H_0 \\text{ verwerfen}" if p_val < alpha else "H_0 \\text{ beibehalten}"
        )
        reason = (
            f"{p_val:.6f} < {alpha}" if p_val < alpha else rf"{p_val:.6f} \geq {alpha}"
        )

        text_decision = (
            f"Da der p-Wert ({p_val:.5f}) kleiner als das Signifikanzniveau α ({alpha}) ist, "
            "verwerfen wir die Nullhypothese. Es gibt signifikante Beweise für H1."
            if p_val < alpha
            else f"Da der p-Wert ({p_val:.5f}) größer als das Signifikanzniveau α ({alpha}) ist, "
            "können wir die Nullhypothese nicht verwerfen. Das Ergebnis ist nicht signifikant."
        )

        steps.append(
            CalculationStep(
                description="Statistische Entscheidung:",
                latex=r"\text{Entscheidung: } "
                + decision
                + r" \quad (\text{da } "
                + reason
                + r")",
            )
        )

        # --- Plot Daten Vorbereitung ---
        # Wir brauchen die t-Verteilung und die markierten Bereiche (kritische Region)
        x_axis = np.linspace(-4, 4, 500)
        # Wenn t-Wert sehr groß ist, Achse erweitern
        limit = max(4, abs(t_val) + 1)
        x_axis = np.linspace(-limit, limit, 500)
        y_axis = t.pdf(x_axis, df)

        # Kritische Grenzen berechnen (für die rote Färbung im Plot)
        crit_region = []  # Liste von Tupeln (start, end)

        if alternative == "two-sided":
            crit_upper = t.ppf(1 - alpha / 2, df)
            crit_lower = -crit_upper
            crit_region = [(-limit, crit_lower), (crit_upper, limit)]
        elif alternative == "greater":
            crit_upper = t.ppf(1 - alpha, df)
            crit_region = [(crit_upper, limit)]
        else:  # less
            crit_lower = t.ppf(alpha, df)
            crit_region = [(-limit, crit_lower)]

        plot_data = {
            "x": x_axis,
            "y": y_axis,
            "highlight_x": t_val,  # Der berechnete t-Wert (Linie)
            "highlight_y": t.pdf(t_val, df),
            "critical_regions": crit_region,  # Bereiche zum Rot markieren
            "xlabel": "t-Wert",
            "ylabel": "Dichte",
            "title": f"t-Verteilung (df={df})",
        }

        return CalculationResult(
            result=p_val,
            steps=steps,
            plot_data=plot_data,
            meta_data={"decision_text": text_decision},
        )


class ChiSquareVarianceTest(StatisticalModel):
    def __init__(self):
        super().__init__("Chi-Quadrat Varianztest (1 Stichprobe)")

        # Symbole
        self.n, self.s_sq, self.sigma_0_sq = sp.symbols(r"n s^2 \sigma_0^2")
        self.chi_sq = sp.symbols(r"\chi^2")

        # Formel: chi^2 = (n-1)*s^2 / sigma_0^2
        self.expr = (self.n - 1) * self.s_sq / self.sigma_0_sq

    def get_variables(self):
        return [
            {"name": "s", "label": "Stichproben-StdAbw (s)", "default": 0.8},
            {"name": "n", "label": "Stichprobengröße (n)", "default": 30, "step": 1},
            {"name": "sigma_0", "label": "Hypothese StdAbw (σ₀)", "default": 1.0},
            {"name": "alpha", "label": "Signifikanz (α)", "default": 0.05},
            {
                "name": "alternative",
                "label": "Alternative H₁",
                "default": "less",  # Default "less" für "konstanter" (geringere Varianz)
                "type": "dropdown",
                "options": [
                    ("Kleiner (<) - Konstanter", "less"),
                    ("Größer (>)", "greater"),
                    ("Ungleich (≠)", "two-sided"),
                ],
            },
        ]

    def get_formula_latex(self) -> str:
        return r"\chi^2 = \frac{(n-1)s^2}{\sigma_0^2}"

    def calculate(
        self, s: float, n: int, sigma_0: float, alpha: float, alternative: str
    ) -> CalculationResult:
        n = int(n)
        s_sq = s**2
        sigma_0_sq = sigma_0**2
        df = n - 1

        steps = []

        # 1. Hypothesen
        if alternative == "less":
            h0 = r"H_0: \sigma^2 \geq " + f"{sigma_0_sq:.2f}"
            h1 = r"H_1: \sigma^2 < " + f"{sigma_0_sq:.2f}"
        elif alternative == "greater":
            h0 = r"H_0: \sigma^2 \leq " + f"{sigma_0_sq:.2f}"
            h1 = r"H_1: \sigma^2 > " + f"{sigma_0_sq:.2f}"
        else:
            h0 = r"H_0: \sigma^2 = " + f"{sigma_0_sq:.2f}"
            h1 = r"H_1: \sigma^2 \neq " + f"{sigma_0_sq:.2f}"

        steps.append(
            CalculationStep(
                "Hypothesen (Varianztest):", f"{h0} \\quad \\text{{vs.}} \\quad {h1}"
            )
        )

        # 2. Teststatistik
        chi_val = (df * s_sq) / sigma_0_sq

        sub_latex = (
            r"\chi^2 = \frac{("
            + f"{n}"
            + r"-1) \cdot "
            + f"{s:.4f}"
            + r"^2}{"
            + f"{sigma_0:.4f}"
            + r"^2} = "
            r"\frac{" + f"{df} \cdot {s_sq:.4f}" + r"}{" + f"{sigma_0_sq:.4f}" + r"}"
        )
        steps.append(
            CalculationStep(
                "Berechnung der Teststatistik:", sub_latex + r" = " + f"{chi_val:.4f}"
            )
        )

        # 3. P-Wert und Kritische Werte
        crit_region = []
        limit = max(chi_val * 1.5, df * 2)  # Plot Limit

        if alternative == "less":
            # Linksseitig
            p_val = chi2.cdf(chi_val, df)
            crit_val = chi2.ppf(alpha, df)
            crit_region = [(0, crit_val)]
            decision_text = (
                f"p={p_val:.5f} < α={alpha}"
                if p_val < alpha
                else f"p={p_val:.5f} >= α={alpha}"
            )

        elif alternative == "greater":
            # Rechtsseitig
            p_val = 1 - chi2.cdf(chi_val, df)
            crit_val = chi2.ppf(1 - alpha, df)
            crit_region = [(crit_val, limit)]
            decision_text = (
                f"p={p_val:.5f} < α={alpha}"
                if p_val < alpha
                else f"p={p_val:.5f} >= α={alpha}"
            )

        else:
            # Zweiseitig
            p_val = 2 * min(chi2.cdf(chi_val, df), 1 - chi2.cdf(chi_val, df))
            crit_lower = chi2.ppf(alpha / 2, df)
            crit_upper = chi2.ppf(1 - alpha / 2, df)
            crit_region = [(0, crit_lower), (crit_upper, limit)]
            decision_text = (
                f"p={p_val:.5f} < α={alpha}"
                if p_val < alpha
                else f"p={p_val:.5f} >= α={alpha}"
            )

        steps.append(CalculationStep("p-Wert Berechnung:", r"p = " + f"{p_val:.6f}"))

        decision = (
            "H_0 \\text{ verwerfen}" if p_val < alpha else "H_0 \\text{ beibehalten}"
        )
        steps.append(
            CalculationStep(
                "Entscheidung:",
                r"\text{Ergebnis: } " + decision + r" \quad (" + decision_text + r")",
            )
        )

        # Plot
        x_axis = np.linspace(0, limit, 500)
        y_axis = chi2.pdf(x_axis, df)

        plot_data = {
            "x": x_axis,
            "y": y_axis,
            "highlight_x": chi_val,
            "highlight_y": chi2.pdf(chi_val, df),
            "critical_regions": crit_region,
            "xlabel": r"$\chi^2$-Wert",
            "ylabel": "Dichte",
            "title": f"Chi-Quadrat Test (df={df})",
        }

        return CalculationResult(p_val, steps, plot_data)


class TwoSampleFTest(StatisticalModel):
    def __init__(self):
        super().__init__("F-Test (Varianzvergleich)")
        self.s1_sq, self.s2_sq = sp.symbols("s_1^2 s_2^2")
        self.expr = self.s1_sq / self.s2_sq

    def get_variables(self):
        return [
            {"name": "s1", "label": "StdAbw. Gruppe 1 (s₁)", "default": 1.2},
            {"name": "n1", "label": "Größe Gruppe 1 (n₁)", "default": 20, "step": 1},
            {"name": "s2", "label": "StdAbw. Gruppe 2 (s₂)", "default": 1.0},
            {"name": "n2", "label": "Größe Gruppe 2 (n₂)", "default": 20, "step": 1},
            {"name": "alpha", "label": "Signifikanz (α)", "default": 0.05},
            {
                "name": "alternative",
                "label": "Alternative H₁",
                "default": "two-sided",
                "type": "dropdown",
                "options": [("Ungleich (≠)", "two-sided"), ("s₁ > s₂", "greater")],
            },
        ]

    def get_formula_latex(self) -> str:
        return r"F = \frac{s_1^2}{s_2^2}"

    def calculate(
        self, s1: float, n1: int, s2: float, n2: int, alpha: float, alternative: str
    ) -> CalculationResult:
        n1, n2 = int(n1), int(n2)
        df1, df2 = n1 - 1, n2 - 1

        # F-Test Logik: F ist meist s1^2 / s2^2
        f_val = (s1**2) / (s2**2)

        steps = []
        steps.append(
            CalculationStep(
                "Berechnung der F-Statistik:",
                r"F = \frac{"
                + f"{s1}^2"
                + r"}{"
                + f"{s2}^2"
                + r"} = "
                + f"{f_val:.4f}",
            )
        )

        # P-Wert
        if alternative == "two-sided":
            # Zweiseitig ist tricky bei F, meist nimmt man 2 * min(P(>F), P(<F))
            p_upper = 1 - f.cdf(f_val, df1, df2)
            p_lower = f.cdf(f_val, df1, df2)
            p_val = 2 * min(p_upper, p_lower)
            steps.append(
                CalculationStep("Zweiseitiger p-Wert:", r"p = " + f"{p_val:.6f}")
            )

            # Kritische Werte für Plot
            crit_upper = f.ppf(1 - alpha / 2, df1, df2)
            crit_lower = f.ppf(alpha / 2, df1, df2)
            limit = max(5, crit_upper * 1.5, f_val * 1.5)
            crit_region = [(0, crit_lower), (crit_upper, limit)]

        else:  # greater
            p_val = 1 - f.cdf(f_val, df1, df2)
            steps.append(
                CalculationStep(
                    "Einseitiger p-Wert (H1: s1 > s2):", r"p = " + f"{p_val:.6f}"
                )
            )

            crit_val = f.ppf(1 - alpha, df1, df2)
            limit = max(5, crit_val * 1.5, f_val * 1.5)
            crit_region = [(crit_val, limit)]

        decision = (
            "H_0 \\text{ verwerfen}" if p_val < alpha else "H_0 \\text{ beibehalten}"
        )
        steps.append(CalculationStep("Entscheidung:", r"\text{Ergebnis: } " + decision))

        # Plot
        x_axis = np.linspace(0.01, limit, 500)
        y_axis = f.pdf(x_axis, df1, df2)

        plot_data = {
            "x": x_axis,
            "y": y_axis,
            "highlight_x": f_val,
            "highlight_y": f.pdf(f_val, df1, df2),
            "critical_regions": crit_region,
            "xlabel": "F-Wert",
            "ylabel": "Dichte",
            "title": f"F-Test (df1={df1}, df2={df2})",
        }
        return CalculationResult(p_val, steps, plot_data)
