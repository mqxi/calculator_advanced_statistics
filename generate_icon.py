# generate_icon.py
import os

import matplotlib.pyplot as plt
import numpy as np

# Pfad sicherstellen
os.makedirs("src/assets", exist_ok=True)

# Icon Design (Modernes Dark-Theme Icon)
# Quadratisch, dunkelgrauer Hintergrund, blau/lila Sigma
fig = plt.figure(figsize=(1, 1), dpi=256)  # 256x256 Pixel
fig.patch.set_facecolor("#18181b")  # Zinc-950 (Hintergrund)

ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()

# Zeichne ein stilisiertes "S" oder Kurve
t = np.linspace(-3, 3, 100)
s = np.exp(-(t**2)) * 0.8  # Gauß-Kurve
ax.plot(t, s + 0.1, color="#6366f1", linewidth=15)  # Indigo Kurve
ax.fill_between(t, s + 0.1, 0, color="#6366f1", alpha=0.3)

# Speichern
plt.savefig("src/assets/icon.png", facecolor=fig.get_facecolor(), dpi=256)
print("Icon erstellt: src/assets/icon.png")
