import ctypes
import os
import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from src.models.bayes import BetaBinomialBayes, GammaGammaBayes, GammaPoissonBayes
from src.models.custom_analysis import CustomContinuousDistribution

# Importiere alle Modelle (Das ist die "Registry" deiner Features)
from src.models.distributions import (
    BernoulliDistribution,
    BinomialDistribution,
    GeometricDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
    WeibullDistribution,
)
from src.models.hypothesis import ChiSquareVarianceTest, OneSampleTTest, TwoSampleFTest
from src.models.mle import MaximumLikelihoodEstimation
from src.models.regression import PolynomialRegression

# Importiere View
from src.ui.main_window import MainWindow


class MainController:
    """
    Der Controller initialisiert die Anwendung, lädt die Modelle
    und verbindet sie mit der Ansicht (View).
    """

    def __init__(self):
        # --- 1. Windows Taskleisten Fix ---
        # Windows gruppiert Python-Skripte oft zusammen. Das hier trennt deine App ab.
        if os.name == "nt":
            myappid = "statcalcpro.advanced.statistics.1.0"  # Beliebige einzigartige ID
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except ImportError:
                pass

        # --- 2. App initialisieren ---
        self.app = QApplication(sys.argv)

        # --- 3. Icon setzen ---
        # Pfad zum Icon bauen (relativ zur Datei main_controller.py)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icon_path = os.path.join(base_dir, "assets", "icon.png")

        if os.path.exists(icon_path):
            self.app.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Warnung: Icon nicht gefunden unter {icon_path}")

        # Style laden (falls vorhanden)
        self._load_styles()

        # 2. View erstellen
        self.view = MainWindow()

        # REMOVED: Controller injection of models. 
        # MainWindow now handles its own model registration to support Categories (TreeWidget).
        
        # REMOVED: self._refresh_view_sidebar() call.

    def _load_styles(self):
        """Versucht, das Stylesheet zu laden."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # styles.qss liegt in src/ui/styles.qss
        style_path = os.path.join(base_dir, "ui", "styles.qss")
        
        try:
            with open(style_path, "r") as f:
                self.app.setStyleSheet(f.read())
            print(f"Info: Stylesheet geladen von {style_path}")
        except FileNotFoundError:
            print(f"CRITICAL WARNUNG: styles.qss nicht gefunden unter {style_path}. Starte mit Standard-Design.")
        except Exception as e:
            print(f"Fehler beim Laden des Stylesheets: {e}")

    def _refresh_view_sidebar(self):
        """
        Leert die Sidebar der View und füllt sie neu mit den Modellen des Controllers.
        Dies stellt sicher, dass View und Controller synchron sind.
        
        DEPRECATED: MainWindow handles TreeWidget population directly now.
        """
        pass

    def run(self):
        """Startet die Event-Loop der Anwendung."""
        self.view.show()
        sys.exit(self.app.exec())
