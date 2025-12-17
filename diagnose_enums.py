from PyQt6.QtWidgets import QAbstractScrollArea, QAbstractItemView, QScrollArea, QApplication
import sys

app = QApplication(sys.argv)

print("QAbstractScrollArea:", dir(QAbstractScrollArea))
print("Has setVerticalScrollMode:", hasattr(QAbstractScrollArea, 'setVerticalScrollMode'))

try:
    print("QAbstractScrollArea.ScrollMode:", QAbstractScrollArea.ScrollMode)
    print("QAbstractScrollArea.ScrollMode.ScrollPerPixel:", QAbstractScrollArea.ScrollMode.ScrollPerPixel)
except AttributeError as e:
    print("Error accessing QAbstractScrollArea.ScrollMode:", e)

try:
    print("QAbstractItemView.ScrollMode:", QAbstractItemView.ScrollMode)
    print("QAbstractItemView.ScrollMode.ScrollPerPixel:", QAbstractItemView.ScrollMode.ScrollPerPixel)
except AttributeError as e:
    print("Error accessing QAbstractItemView.ScrollMode:", e)
