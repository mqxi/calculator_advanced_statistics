import numpy as np
import sympy as sp

from .base import CalculationResult, CalculationStep, StatisticalModel


class PolynomialRegression(StatisticalModel):
    def __init__(self):
        super().__init__("Polynom-Regression (OLS/Ridge)")

        # Symbole für die Anzeige (bleiben allgemein)
        self.lambda_sym = sp.Symbol(r"\lambda")

    def get_variables(self):
        return [
            {
                "name": "x_vals",
                "label": "X Werte (kommagetrennt)",
                "default": "1, 2, 3, 4, 5, 6",
                "type": "text",
            },
            {
                "name": "y_vals",
                "label": "Y Werte (kommagetrennt)",
                "default": "2.1, 3.9, 8.2, 15.5, 26.8, 40.1",
                "type": "text",
            },
            {
                "name": "degree",
                "label": "Polynom-Grad (d)",
                "default": 2,
                "min": 1,
                "max": 20,
                "step": 1,
            },
            {
                "name": "method",
                "label": "Methode",
                "default": "OLS",
                "type": "dropdown",
                "options": [("OLS (Standard)", "OLS"), ("Ridge (L2)", "Ridge")],
            },
            {
                "name": "alpha",
                "label": "Lambda (nur Ridge)",
                "default": 1.0,
                "min": 0.0,
            },
        ]

    def get_formula_latex(self) -> str:
        return r"\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_n x^n"

    def get_info_text(self) -> str:
        return (
            "<html>"
            "Dient zur Modellierung nicht-linearer Zusammenhänge.<br>"
            "• <b>Grad (n)</b>: Bestimmt die Komplexität (1=Linear, 2=Quadratisch, ...).<br>"
            "• <b>Ridge-Alpha</b>: Regularisierungsparameter zur Vermeidung von Overfitting (0 = OLS/Standard).<br><br>"
            "Die Koeffizienten <i>&beta;&#770;</i> werden mittels der Normalengleichung bestimmt:<br>"
            "<i>&beta;&#770; = (X<sup>T</sup> X)<sup>-1</sup> X<sup>T</sup> y</i><br>"
            "Für Ridge-Regression wird ein Regularisierungsterm hinzugefügt:<br>"
            "<i>&beta;&#770; = (X<sup>T</sup> X + &lambda; I)<sup>-1</sup> X<sup>T</sup> y</i>"
            "</html>"
        )

    def _matrix_to_latex(self, mat, max_rows=6, max_cols=6):
        """
        Hilfsfunktion: Wandelt Numpy-Matrizen in Text-Arrays um [ ... ].
        Kürzt ab, wenn die Matrix zu riesig wird (wichtig bei Grad 12!).
        """
        rows, cols = mat.shape

        # Wenn Matrix zu groß, zeigen wir nur die Dimensionen an
        if rows > max_rows or cols > max_cols:
            return r"\text{Matrix } (" + f"{rows} \\times {cols}" + r")"

        rows_str = []
        for row in mat:
            vals = [f"{x:.2f}" for x in row]
            rows_str.append(r"[" + ", ".join(vals) + r"]")

        content = ", ".join(rows_str)
        return r"\left[ " + content + r" \right]"

    def calculate(
        self, x_vals: str, y_vals: str, degree: int, method: str, alpha: float
    ) -> CalculationResult:
        steps = []
        degree = int(degree)

        # 1. Daten parsen
        try:
            x_data = np.array([float(x.strip()) for x in x_vals.split(",")])
            y_data = np.array([float(y.strip()) for y in y_vals.split(",")])
        except ValueError:
            return CalculationResult(
                0, [CalculationStep("Fehler", "Bitte gültige Zahlen eingeben.")]
            )

        if len(x_data) != len(y_data):
            return CalculationResult(
                0, [CalculationStep("Fehler", "X und Y müssen gleich lang sein.")]
            )

        n = len(x_data)

        # 2. Design-Matrix X aufbauen (Vandermonde-Matrix)
        # Spalten: [1, x, x^2, ..., x^degree]
        # Wir bauen das manuell, um die Reihenfolge sicherzustellen (1 links)
        X_cols = []
        for d in range(degree + 1):
            X_cols.append(x_data**d)

        X = np.column_stack(X_cols)
        y = y_data.reshape(-1, 1)

        # Schritt 1: Formel anzeigen
        if method == "Ridge":
            current_formula = r"\hat{\beta} = (X^T X + \lambda I)^{-1} X^T y"
            desc = f"Ridge Regression Formel (mit $\\lambda={alpha}$):"
        else:
            current_formula = r"\hat{\beta} = (X^T X)^{-1} X^T y"
            desc = "OLS Normalengleichung:"

        steps.append(CalculationStep(description=desc, latex=current_formula))

        # Schritt 2: Matrix X und y anzeigen
        # Wir zeigen X nur, wenn es nicht zu riesig ist, sonst nur Info
        x_desc = (
            f"Design-Matrix X (Grad {degree}, Dimension {n}x{degree + 1}) und Vektor y:"
        )
        steps.append(
            CalculationStep(
                description=x_desc,
                latex=r"X = "
                + self._matrix_to_latex(X)
                + r", \quad y = "
                + self._matrix_to_latex(y),
            )
        )

        # Schritt 3: X^T X berechnen
        XtX = X.T @ X
        step3_latex = r"X^T X = " + self._matrix_to_latex(XtX)

        if method == "Ridge":
            # Ridge Regularisierung: Addiere lambda auf die Diagonale (außer Intercept meistens)
            # Hier: Standard Ridge auf alles (wie oft in Lehrbüchern vereinfacht) oder
            # advanced: intercept (index 0) auslassen.
            # Wir machen Standard Ridge auf der gesamten Diagonale für Konsistenz mit der Formel.
            I = np.eye(degree + 1)
            XtX_ridge = XtX + alpha * I

            step3_latex += (
                r" \quad \xrightarrow{+ \lambda I} \quad "
                + self._matrix_to_latex(XtX_ridge)
            )
            XtX = XtX_ridge  # Mit Regularisierung weiterrechnen

        steps.append(
            CalculationStep(
                description=r"Berechnung der Gram-Matrix $X^T X$:", latex=step3_latex
            )
        )

        # Schritt 4: Inversen berechnen
        try:
            XtX_inv = np.linalg.inv(XtX)
            steps.append(
                CalculationStep(
                    description=r"Invertierung $(X^T X)^{-1}$:",
                    latex=r"(...)^{-1} = " + self._matrix_to_latex(XtX_inv),
                )
            )
        except np.linalg.LinAlgError:
            return CalculationResult(
                0,
                [
                    CalculationStep(
                        "Fehler",
                        "Matrix ist singulär (nicht invertierbar). Bei hohem Polynomgrad oft numerisches Problem.",
                    )
                ],
            )

        # Schritt 5: X^T * y
        XtY = X.T @ y

        # Schritt 6: Beta berechnen
        beta = XtX_inv @ XtY

        # Ergebnis schön formatieren als Gleichung
        # beta ist ein Vektor [b0, b1, b2, ...]
        coeffs = beta.flatten()

        eq_parts = []
        for i, b in enumerate(coeffs):
            sign = "+" if b >= 0 else "-"
            val = abs(b)
            if i == 0:
                eq_parts.append(f"{b:.4f}")  # Intercept
            elif i == 1:
                eq_parts.append(f"{sign} {val:.4f}x")
            else:
                eq_parts.append(f"{sign} {val:.4f}x^{i}")

        final_eqn = "y = " + " ".join(eq_parts)

        steps.append(
            CalculationStep(
                description=r"Berechnete Koeffizienten $\hat{\beta}$:",
                latex=r"\hat{\beta} = " + self._matrix_to_latex(beta),
            )
        )

        steps.append(
            CalculationStep(
                description="Finale Regressionsfunktion:",
                latex=r"\text{Ergebnis: } " + final_eqn,
            )
        )

        # --- Plot Daten generieren ---
        # Um die Kurve glatt zu zeichnen, brauchen wir viele Punkte
        x_min, x_max = min(x_data), max(x_data)
        margin = (x_max - x_min) * 0.1
        if margin == 0:
            margin = 1.0

        plot_x = np.linspace(x_min - margin, x_max + margin, 200)

        # y für Plot berechnen: y = b0 + b1*x + b2*x^2 ...
        plot_y = np.zeros_like(plot_x)
        for i, b in enumerate(coeffs):
            plot_y += b * (plot_x**i)

        plot_data = {
            "x_scatter": x_data,  # Echte Punkte
            "y_scatter": y_data,
            "x": plot_x,  # Modell-Kurve
            "y": plot_y,
            "xlabel": "X",
            "ylabel": "Y",
            "title": f"Polynom-Regression (Grad {degree})",
        }

        return CalculationResult(
            result=coeffs,
            steps=steps,
            plot_data=plot_data,
            meta_data={"equation": final_eqn},
        )
