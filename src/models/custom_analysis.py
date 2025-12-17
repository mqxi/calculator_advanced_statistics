import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import brentq

from .base import CalculationResult, CalculationStep, StatisticalModel


class CustomContinuousDistribution(StatisticalModel):
    def __init__(self):
        super().__init__("Custom Funktion (Dichte/PDF)")
        self.x_sym = sp.Symbol("x")

    def get_variables(self):
        return [
            # Der User gibt die Formel als Python/SymPy String ein
            {
                "name": "func_str",
                "label": "f(x) (z.B. 0.5 * exp(-0.5*x))",
                "default": "3 * exp(-3*x)",
                "type": "text",
            },
            {"name": "lower_limit", "label": "Definitionsbereich Min", "default": 0.0},
            {"name": "upper_limit", "label": "Definitionsbereich Max", "default": 10.0},
            {
                "name": "calc_prob_from",
                "label": "P( a < X < b ) - Start (a)",
                "default": 1.0,
            },
            {
                "name": "calc_prob_to",
                "label": "P( a < X < b ) - Ende (b)",
                "default": 2.0,
            },
        ]

    def get_formula_latex(self) -> str:
        return r"\int_{-\infty}^{\infty} f(x) dx = 1"

    def calculate(
        self,
        func_str: str,
        lower_limit: float,
        upper_limit: float,
        calc_prob_from: float,
        calc_prob_to: float,
    ) -> CalculationResult:
        steps = []

        # 1. Funktion parsen (String -> SymPy Ausdruck)
        try:
            # Wir erlauben gängige Mathe-Funktionen
            local_dict = {
                "exp": sp.exp,
                "sin": sp.sin,
                "cos": sp.cos,
                "sqrt": sp.sqrt,
                "pi": sp.pi,
            }
            expr = sp.sympify(func_str, locals=local_dict)
        except Exception as e:
            return CalculationResult(
                0, [CalculationStep("Fehler", f"Konnte Funktion nicht lesen: {e}")]
            )

        # LaTeX der Funktion anzeigen
        steps.append(
            CalculationStep(
                description="Eingegebene Dichtefunktion f(x):",
                latex=r"f(x) = " + sp.latex(expr),
            )
        )

        # 2. In Python-Funktion wandeln (für numerische Integration mit SciPy)
        f_num = sp.lambdify(self.x_sym, expr, "numpy")

        # Wrapper, der 0 zurückgibt, wenn außerhalb des Definitionsbereichs
        # (Wichtig, damit Integration nicht crasht oder Unsinn macht)
        def pdf(val):
            if val < lower_limit or val > upper_limit:
                return 0.0
            try:
                return float(f_num(val))
            except:
                return 0.0

        # --- VALIDIERUNG ---
        # Prüfen, ob es eine gültige PDF ist (Integral muss ca. 1 sein)
        total_area, error = quad(pdf, lower_limit, upper_limit)
        steps.append(
            CalculationStep(
                description="Überprüfung der Normierung (Integral über Definitionsbereich):",
                latex=r"\int_{"
                + f"{lower_limit}"
                + r"}^{"
                + f"{upper_limit}"
                + r"} f(x) dx \approx "
                + f"{total_area:.5f}",
            )
        )

        if not (0.95 < total_area < 1.05):
            steps.append(
                CalculationStep(
                    description="WARNUNG:",
                    latex=r"\text{Achtung: Das Integral ist nicht 1. Ergebnisse sind evtl. falsch!}",
                )
            )

        # --- A. ERWARTUNGSWERT E[X] ---
        # E[X] = integral(x * f(x))
        def expected_val_func(x):
            return x * pdf(x)

        mean, _ = quad(expected_val_func, lower_limit, upper_limit)

        steps.append(
            CalculationStep(
                description="Erwartungswert (Mean):",
                latex=r"\mu = E[X] = \int x \cdot f(x) dx = " + f"{mean:.4f}",
            )
        )

        # --- B. VARIANZ Var(X) ---
        # Var(X) = E[X^2] - (E[X])^2
        def second_moment_func(x):
            return (x**2) * pdf(x)

        moment2, _ = quad(second_moment_func, lower_limit, upper_limit)
        variance = moment2 - mean**2
        std_dev = np.sqrt(variance)

        var_latex = (
            r"\sigma^2 = E[X^2] - (E[X])^2 = "
            + f"{moment2:.4f} - {mean:.4f}^2 = {variance:.4f} "
            r"\quad (\sigma = {std_dev:.4f})"
        )
        steps.append(
            CalculationStep("Varianz und Standardabweichung:", latex=var_latex)
        )

        # --- C. WAHRSCHEINLICHKEIT P(a < X < b) ---
        # Nur berechnen, wenn User sinnvolle Grenzen eingegeben hat
        prob_res = 0.0
        if calc_prob_to > calc_prob_from:
            prob_res, _ = quad(pdf, calc_prob_from, calc_prob_to)
            steps.append(
                CalculationStep(
                    description=f"Wahrscheinlichkeit im Intervall [{calc_prob_from}, {calc_prob_to}]:",
                    latex=r"P("
                    + f"{calc_prob_from} < X < {calc_prob_to}"
                    + r") = \int_{a}^{b} f(x) dx = "
                    + f"{prob_res:.6f}",
                )
            )

        # --- D. QUARTILE (Numerisch) ---
        # Wir suchen x, wo CDF(x) = 0.25, 0.5, 0.75
        # CDF(x) = integral_min^x pdf(t) dt
        # Root finding: CDF(x) - target = 0
        def cdf_root_func(x, target):
            val, _ = quad(pdf, lower_limit, x)
            return val - target

        try:
            # Median (0.5)
            # Wir suchen im Bereich [lower, upper]. Brentq ist robust.
            median = brentq(cdf_root_func, lower_limit, upper_limit, args=(0.5,))
            q1 = brentq(
                cdf_root_func, lower_limit, median, args=(0.25,)
            )  # Suche Q1 links vom Median
            q3 = brentq(
                cdf_root_func, median, upper_limit, args=(0.75,)
            )  # Suche Q3 rechts vom Median

            quartile_latex = (
                r"Q_1 (25\%) = " + f"{q1:.3f}" + r", \quad "
                r"Q_2 (\text{Median}) = " + f"{median:.3f}" + r", \quad "
                r"Q_3 (75\%) = " + f"{q3:.3f}"
            )
            steps.append(CalculationStep("Berechnete Quartile:", latex=quartile_latex))
        except Exception:
            steps.append(
                CalculationStep(
                    "Quartile:",
                    r"\text{Konnte numerisch nicht bestimmt werden (Definitionsbereich prüfen).}",
                )
            )

        # --- PLOT ---
        # Wir plotten die Funktion und schattieren den Wahrscheinlichkeitsbereich
        x_plot = np.linspace(lower_limit, upper_limit, 400)
        # Vektorisiertes Aufrufen von pdf für Performance
        y_plot = np.array([pdf(xi) for xi in x_plot])

        plot_data = {
            "x": x_plot,
            "y": y_plot,
            "highlight_x": mean,  # Mean als Linie
            "highlight_y": pdf(mean),
            # Wir nutzen "critical_regions" hier missbräuchlich für die Schattierung des gesuchten Bereichs
            "critical_regions": [(calc_prob_from, calc_prob_to)],
            "xlabel": "x",
            "ylabel": "f(x)",
            "title": f"Custom PDF: {func_str}",
        }

        return CalculationResult(prob_res, steps, plot_data)
