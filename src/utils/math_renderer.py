import io

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PyQt6.QtGui import QImage, QPixmap

# Wir setzen das Backend auf 'Agg', damit Matplotlib keine Fenster öffnet,
# sondern nur im Hintergrund Bilder berechnet.
matplotlib.use("Agg")


def render_latex_to_pixmap(latex_str, font_size=12, dpi=120, color="black"):
    """
    Rendert einen LaTeX-String in eine QPixmap für PyQt.

    Args:
        latex_str (str): Der LaTeX-Code (z.B. r"\frac{x}{y}").
        font_size (int): Schriftgröße.
        dpi (int): Auflösung (höher = schärfer).
        color (str): Textfarbe (z.B. 'black', '#333333').

    Returns:
        QPixmap: Das gerenderte Bild (transparent).
    """

    # Sicherstellen, dass der String im Math-Mode ist, falls noch nicht geschehen
    # Matplotlib braucht $...$ für Formeln, wenn es reiner Text ist.
    # Unsere Models liefern meist reine Formeln, aber sicher ist sicher.
    clean_latex = latex_str.strip()
    if not clean_latex.startswith("$"):
        clean_latex = f"${clean_latex}$"

    # 1. Matplotlib Figure erstellen (sehr klein, da wir 'bbox_inches=tight' nutzen)
    fig = Figure(figsize=(0.1, 0.1), dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    # 2. Text rendern
    # Wir platzieren den Text. Die Koordinaten sind fast egal,
    # da wir später zuschneiden.
    text_elem = fig.text(0, 0, clean_latex, fontsize=font_size, color=color)

    # Achsen entfernen (wir wollen nur den Text)
    # Da wir fig.text nutzen, gibt es eh keine Achsen, aber wir müssen sicherstellen,
    # dass kein Hintergrund gezeichnet wird.
    fig.patch.set_alpha(0)  # Transparenter Hintergrund

    # 3. Buffer erstellen
    buf = io.BytesIO()

    try:
        # 4. Speichern (Rendern)
        # bbox_inches='tight' schneidet den weißen Rand weg.
        # pad_inches=0.05 sorgt für minimalen Abstand, damit kursive Buchstaben (f, j) nicht abgeschnitten werden.
        fig.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=0.05,
            transparent=True,
            bbox_extra_artists=[
                text_elem
            ],  # Wichtig: Damit Matplotlib weiß, was drauf sein MUSS
        )

        # 5. Buffer in QPixmap umwandeln
        buf.seek(0)
        qimg = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(qimg)

        # Aufräumen (Wichtig für Speicher bei vielen Formeln!)
        plt.close(fig)
        buf.close()

        return pixmap

    except Exception as e:
        # Fallback: Wenn LaTeX falsch ist (z.B. Syntaxfehler), geben wir Text zurück
        # oder ein leeres Bild, damit die App nicht abstürzt.
        print(f"LaTeX Rendering Fehler: {e} | String: {latex_str}")
        return None
