# StatCalcPro - Advanced Statistics Calculator

StatCalcPro ist eine Python-basierte Desktop-Anwendung für fortgeschrittene statistische Berechnungen und Datenvisualisierung. Die Anwendung nutzt `PyQt6` für das grafische Interface und wissenschaftliche Bibliotheken wie `NumPy`, `SciPy` und `SymPy` für mathematische Operationen.

## Features

Die Anwendung bietet eine Sammlung statistischer Modelle, organisiert in kategorisierte Module:

*   **Regression**: Polynom-Regression (OLS/Ridge) mit interaktiven Plots.
*   **Verteilungen**: Berechnung von Dichtefunktionen (PDF) und Wahrscheinlichkeiten für diverse Verteilungen (Normal, Exponential, Binomial, Poisson, Weibull, etc.).
*   **Hypothesentests**: Parametrische und nicht-parametrische Tests (T-Test, Chi-Quadrat, F-Test).
*   **Bayes Statistik**: Konjugierte Priors und Posterior-Updates (Beta-Binomial, Gamma-Poisson).
*   **Schätzung**: Maximum Likelihood Estimation (MLE) für benutzerdefinierte Daten.
*   **Benutzerdefiniert**: Definition eigener Dichtefunktionen.

## Projektstruktur

Der Quellcode befindet sich im `src`-Verzeichnis:

```text
calculator_advanced_statistics/
├── main.py                  # Einstiegspunkt der Anwendung
├── README.md                # Projektdokumentation
├── src/
│   ├── assets/              # Statische Ressourcen (Icons, SVGs)
│   ├── controllers/
│   │   └── main_controller.py  # Initialisierung und App-Startlogik
│   ├── models/              # Mathematische Implementierungen
│   │   ├── bayes.py         # Bayes-Update Logik
│   │   ├── custom_analysis.py # Benutzerdefinierte Parser
│   │   ├── distributions.py # Verteilungsmodelle (Normal, Expo, etc.)
│   │   ├── hypothesis.py    # Statistische Tests
│   │   ├── mle.py          # Numerische Optimierung
│   │   └── regression.py    # Regressionsanalysen
│   └── ui/                  # Grafische Benutzeroberfläche
│       ├── main_window.py   # Hauptfenster & Navigationslogik (QTreeWidget)
│       ├── styles.qss       # Design-Definitionen (CSS-ähnlich)
│       └── widgets/         # Wiederverwendbare UI-Komponenten
│           ├── latex_label.py # Rendering von LaTeX-Formeln
│           └── plot_canvas.py # Matplotlib-Integration
```

## Installation & Ausführung

### Voraussetzungen
*   Python 3.10+
*   Abhängigkeiten: `PyQt6`, `numpy`, `scipy`, `matplotlib`, `sympy`

### Starten
Das Programm kann über den Einstiegspunkt `main.py` gestartet werden:

```bash
# Mit uv (empfohlen)
uv run main.py

# Oder direkt mit Python
python main.py
```
