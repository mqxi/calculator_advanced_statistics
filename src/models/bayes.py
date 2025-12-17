import numpy as np
import sympy as sp
from scipy.stats import beta, gamma

from .base import CalculationResult, CalculationStep, StatisticalModel


class BetaBinomialBayes(StatisticalModel):
    def __init__(self):
        super().__init__("Bayes Update: Beta-Binomial")

        # Symbole
        self.p = sp.Symbol("p")
        self.alpha, self.beta = sp.Symbol(r"\alpha"), sp.Symbol(r"\beta")
        self.k, self.n = sp.Symbol("k"), sp.Symbol("n")

        # Bayes Theorem Proportionalität
        # Posterior ~ Likelihood * Prior
        self.prior_expr = self.p ** (self.alpha - 1) * (1 - self.p) ** (self.beta - 1)
        self.likelihood_expr = self.p**self.k * (1 - self.p) ** (self.n - self.k)

    def get_variables(self):
        return [
            {
                "name": "alpha_prior",
                "label": "Prior α (Erfolge)",
                "default": 1.0,
                "min": 0.1,
            },
            {
                "name": "beta_prior",
                "label": "Prior β (Misserfolge)",
                "default": 1.0,
                "min": 0.1,
            },
            {
                "name": "n_trials",
                "label": "Neue Versuche (n)",
                "default": 10,
                "step": 1,
            },
            {"name": "k_success", "label": "Neue Erfolge (k)", "default": 7, "step": 1},
        ]

    def get_formula_latex(self) -> str:
        return r"Posterior \propto Likelihood \cdot Prior"

    def calculate(
        self, alpha_prior: float, beta_prior: float, n_trials: int, k_success: int
    ) -> CalculationResult:
        n_trials = int(n_trials)
        k_success = int(k_success)

        # Sicherheitscheck
        if k_success > n_trials:
            return CalculationResult(
                0, [CalculationStep("Fehler", "k kann nicht größer als n sein.")]
            )

        steps = []

        # 1. Prior Darstellung
        steps.append(
            CalculationStep(
                description="1. Der Prior (Vorwissen) folgt einer Beta-Verteilung:",
                latex=r"\text{Prior} \sim Beta(\alpha="
                + f"{alpha_prior}, "
                + r"\beta="
                + f"{beta_prior})",
            )
        )

        # 2. Likelihood
        steps.append(
            CalculationStep(
                description=f"2. Die Daten ({k_success} Erfolge in {n_trials} Versuchen) liefern die Likelihood:",
                latex=r"Likelihood \propto p^{"
                + f"{k_success}"
                + r"} (1-p)^{"
                + f"{n_trials - k_success}"
                + r"}",
            )
        )

        # 3. Update Regeln
        alpha_post = alpha_prior + k_success
        beta_post = beta_prior + (n_trials - k_success)

        update_latex = (
            r"\alpha_{post} = "
            + f"{alpha_prior} + {k_success} = {alpha_post}"
            + r", \quad "
            r"\beta_{post} = " + f"{beta_prior} + {n_trials - k_success} = {beta_post}"
        )

        steps.append(
            CalculationStep(
                description="3. Berechnung der Posterior-Parameter (Konjugiertes Update):",
                latex=update_latex,
            )
        )

        # 4. Posterior Ergebnis
        mean_post = alpha_post / (alpha_post + beta_post)

        steps.append(
            CalculationStep(
                description=f"4. Das neue Wissen (Posterior) mit Erwartungswert E[p] = {mean_post:.3f}:",
                latex=r"\text{Posterior} \sim Beta(" + f"{alpha_post}, {beta_post})",
            )
        )

        # Plot Daten erstellen
        x = np.linspace(0, 1, 300)
        y_prior = beta.pdf(x, alpha_prior, beta_prior)
        y_post = beta.pdf(x, alpha_post, beta_post)

        # Likelihood skalieren, damit man sie im gleichen Plot sieht (nur Form ist wichtig)
        # Wir nutzen die PDF der Beta-Verteilung, die der Likelihood entsprechen würde (k+1, n-k+1)
        # um die Kurve schön darzustellen, auch wenn Likelihood mathematisch keine PDF ist.
        y_likelihood = beta.pdf(x, k_success + 1, n_trials - k_success + 1)

        plot_data = {
            "x": x,
            "y_prior": y_prior,
            "y_post": y_post,
            "y_likelihood": y_likelihood,  # Optional zu zeichnen
            "xlabel": "Wahrscheinlichkeit p",
            "ylabel": "Dichte",
            "title": "Bayesianisches Update (Beta-Binomial)",
        }

        return CalculationResult(
            result=mean_post,
            steps=steps,
            plot_data=plot_data,
            meta_data={
                "type": "bayesian_plot"
            },  # Hinweis für GUI, dass hier 3 Linien geplottet werden
        )


class GammaPoissonBayes(StatisticalModel):
    def __init__(self):
        super().__init__("Bayes Update: Gamma-Poisson")
        # Wir nutzen die Parametrisierung Shape (k oder alpha) und Rate (beta)
        # E[x] = alpha / beta

    def get_variables(self):
        return [
            {"name": "alpha_prior", "label": "Prior Shape (α)", "default": 2.0},
            {"name": "beta_prior", "label": "Prior Rate (β)", "default": 1.0},
            {"name": "sum_x", "label": "Summe Beobachtungen (Σx)", "default": 10.0},
            {"name": "n_obs", "label": "Anzahl Beobachtungen (n)", "default": 4.0},
        ]

    def get_formula_latex(self) -> str:
        return r"\lambda | x \sim Gamma(\alpha + \sum x_i, \beta + n)"

    def calculate(
        self, alpha_prior: float, beta_prior: float, sum_x: float, n_obs: float
    ) -> CalculationResult:
        steps = []

        # 1. Prior
        mean_prior = alpha_prior / beta_prior
        steps.append(
            CalculationStep(
                description=f"Prior Gamma-Verteilung (Erwartungswert {mean_prior:.2f}):",
                latex=r"\lambda \sim Gamma(\alpha="
                + f"{alpha_prior}, "
                + r"\beta="
                + f"{beta_prior})",
            )
        )

        # 2. Update Regeln
        alpha_post = alpha_prior + sum_x
        beta_post = beta_prior + n_obs

        update_latex = (
            r"\alpha_{post} = "
            + f"{alpha_prior} + {sum_x} = {alpha_post}"
            + r", \quad "
            r"\beta_{post} = " + f"{beta_prior} + {n_obs} = {beta_post}"
        )

        steps.append(
            CalculationStep(
                description="Update der Hyperparameter:", latex=update_latex
            )
        )

        # 3. Posterior Mean
        mean_post = alpha_post / beta_post
        steps.append(
            CalculationStep(
                description=f"Posterior Gamma-Verteilung (Neuer Erwartungswert {mean_post:.2f}):",
                latex=r"\lambda_{new} \sim Gamma(" + f"{alpha_post}, {beta_post})",
            )
        )

        # Plotting
        # Wähle Bereich dynamisch basierend auf dem Mean
        max_x = max(mean_prior, mean_post) * 3
        if max_x == 0:
            max_x = 10

        x = np.linspace(0, max_x, 300)
        # Scipy nutzt (a, scale=1/b) Konvention für Gamma
        y_prior = gamma.pdf(x, a=alpha_prior, scale=1.0 / beta_prior)
        y_post = gamma.pdf(x, a=alpha_post, scale=1.0 / beta_post)

        plot_data = {
            "x": x,
            "y_prior": y_prior,
            "y_post": y_post,
            "xlabel": "Rate λ",
            "ylabel": "Dichte",
            "title": "Bayesianisches Update (Gamma-Poisson)",
        }

        return CalculationResult(
            result=mean_post,
            steps=steps,
            plot_data=plot_data,
            meta_data={"type": "bayesian_plot"},
        )


class GammaGammaBayes(StatisticalModel):
    def __init__(self):
        super().__init__("Bayes Update: Gamma-Gamma (Stetig)")
        # Theta ist hier der Parameter der Verteilung (Rate)
        self.theta = sp.Symbol(r"\theta")

    def get_variables(self):
        return [
            # Prior Parameter
            {"name": "alpha_prior", "label": "Prior Alpha", "default": 3.0},
            {"name": "beta_prior", "label": "Prior Beta (Rate)", "default": 1.0},
            # Likelihood Daten
            {
                "name": "n",
                "label": "Anzahl Beobachtungen (n)",
                "default": 10,
                "step": 1,
            },
            {"name": "mean_x", "label": "Mittelwert der Daten (x_bar)", "default": 4.5},
            # Schätzung
            {
                "name": "loss_type",
                "label": "Schätzer (Loss)",
                "default": "mean",
                "type": "dropdown",
                "options": [
                    ("Quadratisch (Mean)", "mean"),
                    ("0-1 Loss (Mode)", "mode"),
                ],
            },
        ]

    def get_formula_latex(self) -> str:
        return r"\pi(\theta|x) \propto L(\theta) \cdot \pi(\theta)"

    def calculate(
        self,
        alpha_prior: float,
        beta_prior: float,
        n: int,
        mean_x: float,
        loss_type: str,
    ) -> CalculationResult:
        steps = []
        n = int(n)
        sum_x = mean_x * n  # Summe der Daten rekonstruieren

        # 1. Prior anzeigen
        steps.append(
            CalculationStep(
                "1. Prior Verteilung (Gamma):",
                rf"\theta \sim \text{{Gamma}}(\alpha_{{prior}}={alpha_prior}, \beta_{{prior}}={beta_prior})",
            )
        )

        # 2. Likelihood Info
        # Wenn X ~ Gamma(alpha_like, theta), dann ist Conjugate Prior für theta auch Gamma.
        # Update Regeln: alpha_new = alpha_old + n * alpha_like
        # beta_new = beta_old + sum(x)
        # HINWEIS: Aufgabe 6 sagt Likelihood hat alpha=3 (fix).
        alpha_likelihood = 3

        steps.append(
            CalculationStep(
                "2. Likelihood Modell:",
                rf"X_i \sim \text{{Gamma}}(\alpha={alpha_likelihood}, \theta) \quad (\text{{n}}={n}, \sum x_i={sum_x:.2f})",
            )
        )

        # 3. Posterior Parameter berechnen
        alpha_post = alpha_prior + (n * alpha_likelihood)
        beta_post = beta_prior + sum_x

        update_latex = (
            rf"\alpha_{{post}} = \alpha_{{prior}} + n \cdot \alpha_{{L}} = {alpha_prior} + {n}\cdot{alpha_likelihood} = {alpha_post}, \quad "
            rf"\beta_{{post}} = \beta_{{prior}} + \sum x_i = {beta_prior} + {sum_x:.2f} = {beta_post:.2f}"
        )
        steps.append(CalculationStep("3. Posterior Parameter (Update):", update_latex))

        # 4. Punktschätzung
        if loss_type == "mean":
            # Mean einer Gamma(a, b) ist a / b (bei Raten-Parametrisierung)
            est_val = alpha_post / beta_post
            est_desc = "Bayes-Schätzer (Erwartungswert/Quadratischer Verlust):"
            est_formula = r"\hat{\theta} = \frac{\alpha_{post}}{\beta_{post}}"
        else:
            # Mode ist (a - 1) / b
            if alpha_post > 1:
                est_val = (alpha_post - 1) / beta_post
            else:
                est_val = 0  # Randlösung
            est_desc = "Bayes-Schätzer (Modus/MAP):"
            est_formula = r"\hat{\theta} = \frac{\alpha_{post}-1}{\beta_{post}}"

        steps.append(CalculationStep(est_desc, rf"{est_formula} = {est_val:.6f}"))

        # Plot Daten (Prior vs Posterior)
        # Wir plotten Gamma-Dichten
        from scipy.stats import gamma

        # x-Achse dynamisch wählen (wo die Masse liegt)
        mean_post = alpha_post / beta_post
        std_post = np.sqrt(alpha_post / (beta_post**2))
        limit = mean_post + 4 * std_post
        x_axis = np.linspace(0, limit, 400)

        # Scipy gamma nutzt (a, scale=1/beta) wenn beta rate parameter ist!
        y_prior = gamma.pdf(x_axis, a=alpha_prior, scale=1.0 / beta_prior)
        y_post = gamma.pdf(x_axis, a=alpha_post, scale=1.0 / beta_post)

        plot_data = {
            "x": x_axis,
            "y_prior": y_prior,
            "y_post": y_post,
            "highlight_x": est_val,
            "title": "Update von Prior zu Posterior (Gamma-Gamma)",
            "xlabel": r"Parameter $\theta$",
            "ylabel": "Dichte",
        }

        return CalculationResult(
            est_val, steps, plot_data, meta_data={"type": "bayesian_plot"}
        )
