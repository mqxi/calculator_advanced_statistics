from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy

from src.utils.math_renderer import render_latex_to_pixmap


class LatexLabel(QLabel):
    """
    Ein benutzerdefiniertes Widget, das LaTeX-Code als Bild rendert.
    """

    def __init__(
        self, latex_str: str, font_size: int = 14, color: str = "#2d3436", parent=None
    ):
        super().__init__(parent)

        # Konfiguration speichern
        self.latex_str = latex_str
        self.font_size = font_size
        self.text_color = color

        # UI-Einstellungen
        # Wir wollen, dass das Widget linksbündig ist und nicht unnötig Platz frisst
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        # Initial rendern
        self.render()

    def set_latex(self, latex_str: str):
        """Erlaubt das nachträgliche Ändern der Formel."""
        self.latex_str = latex_str
        self.render()

    def render(self):
        """Ruft den Renderer auf und setzt das Pixmap."""
        if not self.latex_str:
            self.clear()
            return

        # Rendern mit unserer Utility-Funktion
        pixmap = render_latex_to_pixmap(
            self.latex_str, font_size=self.font_size, color=self.text_color
        )

        if pixmap:
            self.setPixmap(pixmap)
        else:
            # Fallback: Falls das Rendern fehlschlägt (z.B. Syntaxfehler),
            # zeigen wir den rohen Text an, damit der User sieht, was falsch ist.
            self.setText(self.latex_str)
            self.setStyleSheet("color: red;")  # Rot markieren bei Fehler
