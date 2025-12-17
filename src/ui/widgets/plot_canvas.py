import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Styling für Matplotlib (damit es zum modernen Look passt)
matplotlib.style.use("bmh")  # 'bmh' oder 'ggplot' sehen clean aus


class PlotCanvas(FigureCanvasQTAgg):
    """
    Ein universelles Widget, das Matplotlib-Graphen in PyQt anzeigt.
    Es interpretiert das 'plot_data' Dictionary aus den Models.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Figure erstellen
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)

        # Initialisierung der Elternklasse
        super().__init__(self.fig)
        self.setParent(parent)

        # Standard-Styling des Canvas
        self.fig.patch.set_facecolor("#f4f6f9")  # Hintergrund passend zum GUI
        self.ax.set_facecolor("white")

    def clear(self):
        """Löscht den aktuellen Plot."""
        self.ax.clear()
        self.draw()

    def plot(self, data: dict):
        """
        Hauptfunktion zum Zeichnen.
        Entscheidet anhand der Keys im Dictionary, welcher Plot-Typ erstellt wird.
        """
        self.ax.clear()

        if not data:
            self.draw()
            return

        # --- 1. Spezialfall: Bayesianische Plots (Prior, Posterior, Likelihood) ---
        if "y_prior" in data and "y_post" in data:
            self._plot_bayes(data)

        # --- 2. Spezialfall: Regression (Scatter + Line) ---
        elif "x_scatter" in data:
            self._plot_regression(data)

        # --- 3. Standard: Verteilungen & Hypothesentests ---
        else:
            self._plot_standard_distribution(data)

        # Labels und Titel setzen (falls vorhanden)
        if "xlabel" in data:
            self.ax.set_xlabel(data["xlabel"])
        if "ylabel" in data:
            self.ax.set_ylabel(data["ylabel"])
        if "title" in data:
            self.ax.set_title(data["title"], fontsize=10)

        # Grid und Layout
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

        # Zeichnen auslösen (wichtig!)
        self.draw()

    def _plot_bayes(self, data):
        """Zeichnet Prior, Posterior und ggf. Likelihood."""
        x = data["x"]

        # Labels holen (Standard oder Custom)
        labels = data.get("legend_labels", ["Prior", "Posterior"])
        label_1 = labels[0]
        label_2 = labels[1]

        # Kurve 1 (Grau/Gestrichelt) -> Hier Likelihood
        self.ax.plot(
            x, data["y_prior"], label=label_1, linestyle="--", color="gray", alpha=0.7
        )

        # ... (Likelihood Plot Logik falls vorhanden) ...

        # Kurve 2 (Blau/Fett) -> Hier Log-Likelihood
        self.ax.plot(
            x, data["y_post"], label=label_2, linewidth=2.5, color="#6366f1"
        )  # Indigo Farbe

        self.ax.legend()

    def _plot_regression(self, data):
        """Zeichnet Scatterplot der Daten und Regressionsgerade."""
        # Die echten Datenpunkte
        self.ax.scatter(
            data["x_scatter"],
            data["y_scatter"],
            color="#2d3436",
            alpha=0.7,
            label="Daten",
        )

        # Die berechnete Gerade
        self.ax.plot(data["x"], data["y"], color="#e84393", linewidth=2, label="Modell")

        self.ax.legend()

    def _plot_standard_distribution(self, data):
        """
        Handhabt Normalverteilung, Binomialverteilung und t-Tests.
        """
        # SICHERHEIT: x und y immer in numpy arrays wandeln
        x = np.array(data["x"])
        y = np.array(data["y"])

        # Prüfen, ob wir diskrete Daten haben
        is_discrete = False
        if len(x) < 50 and np.all(np.mod(x, 1) == 0):
            is_discrete = True

        if is_discrete:
            # Balkendiagramm
            bars = self.ax.bar(x, y, color="#74b9ff", alpha=0.6, width=0.6)

            if "highlight_x" in data:
                k = int(data["highlight_x"])
                if k in x:
                    # Index finden
                    idx = np.where(x == k)[0][0]
                    if idx < len(bars):
                        bars[idx].set_color("#d63031")
                        bars[idx].set_alpha(1.0)
        else:
            # Linienplot
            self.ax.plot(x, y, color="#0984e3", linewidth=2)
            self.ax.fill_between(x, y, alpha=0.1, color="#0984e3")

            if "highlight_x" in data:
                hx = data["highlight_x"]
                hy = data.get("highlight_y", 0)
                self.ax.axvline(hx, color="#d63031", linestyle="--", alpha=0.8)
                self.ax.plot(hx, hy, "o", color="#d63031")

            # Kritische Bereiche schattieren
            if "critical_regions" in data:
                for start, end in data["critical_regions"]:
                    # HIER WAR DER FEHLER: x muss ein Array sein für die Bedingung
                    self.ax.fill_between(
                        x,
                        y,
                        where=((x >= start) & (x <= end)),
                        color="#d63031",
                        alpha=0.3,
                        label="Ablehnungsbereich",
                    )

                handles, labels = self.ax.get_legend_handles_labels()
                if labels:
                    self.ax.legend()
