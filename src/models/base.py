from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sympy as sp

# --- DATENSTRUKTUREN ---
# Diese Klassen definieren, wie eine "Antwort" des Rechners aussieht.


@dataclass
class CalculationStep:
    """
    Repräsentiert einen einzelnen Rechenschritt für die Anzeige.

    Attributes:
        description (str): Erklärungstext (z.B. "Einsetzen der Werte in die Formel")
        latex (str): Die mathematische Darstellung (z.B. "f(x) = ...")
    """

    description: str
    latex: str


@dataclass
class CalculationResult:
    """
    Das Endergebnis einer Berechnung, das an das GUI zurückgegeben wird.

    Attributes:
        result (Any): Das numerische Ergebnis (float, array, etc.)
        steps (List[CalculationStep]): Liste der Rechenwege für die Anzeige
        plot_data (Optional[Dict]): Daten für den Plot (x, y Werte), falls vorhanden
        meta_data (Optional[Dict]): Zusätzliche Infos (z.B. p-Value Entscheidung)
    """

    result: Any
    steps: List[CalculationStep] = field(default_factory=list)
    plot_data: Optional[Dict[str, Any]] = None
    meta_data: Optional[Dict[str, Any]] = None


# --- BASISKLASSEN ---


class StatisticalModel(ABC):
    """
    Abstrakte Basisklasse für JEDES statistische Modul
    (Verteilungen, Tests, Regressionen).
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def get_formula_latex(self) -> str:
        """Gibt die allgemeine Formel des Modells als LaTeX-String zurück."""
        pass

    def get_info_text(self) -> str:
        """
        Gibt zusätzlichen Text zu Definitionen, Abhängigkeiten und Beschränkungen zurück.
        Kann Zeilenumbrüche enthalten.
        """
        return "Keine weiteren Informationen verfügbar."

    @abstractmethod
    def calculate(self, **kwargs) -> CalculationResult:
        """
        Führt die Berechnung durch.

        Args:
            **kwargs: Die Eingabeparameter (z.B. mu=0, sigma=1, x=2)

        Returns:
            CalculationResult Objekt mit Wert, Schritten und Plot-Daten.
        """
        pass


class DistributionModel(StatisticalModel):
    """
    Spezifische Basisklasse für Wahrscheinlichkeitsverteilungen.
    """

    def __init__(self, name: str, type_: str):
        """
        Args:
            name: Name der Verteilung (z.B. "Normalverteilung")
            type_: 'continuous' oder 'discrete'
        """
        super().__init__(name)
        self.type = type_  # Wichtig für die Art des Plots (Linie vs. Balken)

    def _substitute_symbols(
        self, expression: sp.Expr, values: Dict[sp.Symbol, float]
    ) -> str:
        """
        Hilfsfunktion, um Symbole in einer Formel durch Zahlen zu ersetzen,
        ohne sie sofort auszurechnen (für den Rechenweg).

        Gibt einen LaTeX-String zurück.
        """
        # Wir ersetzen die Symbole durch eine "unbekannte" Funktion oder Wrapper,
        # damit SymPy sie nicht sofort vereinfacht (z.B. 2+2 zu 4 macht),
        # sondern erst beim Anzeigen.
        # *Einfacherer Ansatz für den Anfang:*
        # Wir iterieren über die Werte und ersetzen im LaTeX-String manuell oder nutzen subs() vorsichtig.

        # Hier nutzen wir das normale subs, aber für die Anzeige runden wir floats
        subbed = expression.subs(values)
        return sp.latex(subbed)

    @abstractmethod
    def get_variables(self) -> List[str]:
        """
        Gibt zurück, welche Eingabefelder das GUI anzeigen muss.
        z.B. ['mu', 'sigma', 'x']
        """
        pass
