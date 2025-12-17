from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QAbstractScrollArea,
    QAbstractItemView,
    QListView,
    QTreeWidget,
    QTreeWidgetItem,
)

from src.ui.widgets.latex_label import LatexLabel
from src.ui.widgets.plot_canvas import PlotCanvas

# Import Models
from src.models.regression import PolynomialRegression
from src.models.distributions import (
    NormalDistribution,
    BinomialDistribution,
    WeibullDistribution,
    PoissonDistribution,
    GeometricDistribution,
    NegativeBinomialDistribution,
    BernoulliDistribution,
    ExponentialDistribution,
)
from src.models.hypothesis import (
    OneSampleTTest,
    ChiSquareVarianceTest,
    TwoSampleFTest,
)
from src.models.mle import MaximumLikelihoodEstimation
from src.models.bayes import (
    BetaBinomialBayes,
    GammaPoissonBayes,
    GammaGammaBayes,
)
from src.models.custom_analysis import CustomContinuousDistribution


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StatCalcPro")
        self.resize(1280, 850)

        self.models = {
            "Regression": [
                PolynomialRegression(),
            ],
            "Verteilungen": [
                NormalDistribution(),
                BinomialDistribution(),
                WeibullDistribution(),
                PoissonDistribution(),
                GeometricDistribution(),
                NegativeBinomialDistribution(),
                BernoulliDistribution(),
                ExponentialDistribution(),
            ],
            "Hypothesentests": [
                OneSampleTTest(),
                ChiSquareVarianceTest(),
                TwoSampleFTest(),
            ],
            "Bayes Statistik": [
                BetaBinomialBayes(),
                GammaPoissonBayes(),
                GammaGammaBayes(),
            ],
            "Schätzung": [
                MaximumLikelihoodEstimation(),
            ],
            "Sonstiges": [
                CustomContinuousDistribution(),
            ]
        }
        self.current_model = None
        self.input_widgets = {}

        # --- HAUPT WIDGET ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Hauptlayout ohne Ränder (damit Splitter den Rand berührt)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 1. SIDEBAR (Links) ---
        self.sidebar = QTreeWidget()
        self.sidebar.setFixedWidth(260)
        self.sidebar.setObjectName("sidebar")  # Für CSS
        self.sidebar.setHeaderHidden(True)
        self.sidebar.setIndentation(15)
        
        # SCROLLBAR FIX: Balken anzeigen, wenn nötig
        self.sidebar.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Smooth Scrolling aktivieren
        self.sidebar.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.sidebar.itemClicked.connect(self.on_model_selected)

        # Populate Sidebar
        print(f"Loading models into sidebar...")
        font_category = self.sidebar.font()
        font_category.setBold(True)
        font_category.setPointSize(15)

        for category, models in self.models.items():
            cat_item = QTreeWidgetItem(self.sidebar)
            cat_item.setText(0, category)
            cat_item.setFont(0, font_category)
            # Make category not selectable, just expandable
            cat_item.setFlags(Qt.ItemFlag.ItemIsEnabled) 
            
            for model in models:
                item = QTreeWidgetItem(cat_item)
                item.setText(0, model.name)
                # Store model object in item data (Key 0, Role UserRole)
                item.setData(0, Qt.ItemDataRole.UserRole, model)
                print(f" - Added: {model.name} to {category}")


        # --- 2. RECHTER BEREICH (Inhalt) ---
        # Dieser Bereich enthält den Toggle-Button UND die ScrollArea
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # --- TOP BAR (Mit Toggle Button) ---
        top_bar = QFrame()
        top_bar.setObjectName("top_bar")  # Für CSS (Hintergrundfarbe wie Main)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(10, 10, 10, 0)  # Etwas Platz

        # Der Toggle Button (☰)
        self.btn_toggle = QPushButton("☰")
        self.btn_toggle.setObjectName("btn_toggle")  # Spezielles CSS
        self.btn_toggle.setFixedSize(40, 40)
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.clicked.connect(self.toggle_sidebar)

        top_bar_layout.addWidget(self.btn_toggle)
        top_bar_layout.addStretch()  # Rest nach rechts schieben

        right_layout.addWidget(top_bar)

        # --- SCROLL AREA (Der eigentliche Inhalt) ---
        self.right_scroll_area = QScrollArea()
        self.right_scroll_area.setWidgetResizable(True)  # WICHTIG gegen Bugs
        self.right_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        # Smooth Scrolling aktivieren
        # self.right_scroll_area.setVerticalScrollMode(1) 
        # Transparenter Hintergrund für Modern Look
        self.right_scroll_area.setStyleSheet("background-color: transparent;")

        # Container für den Inhalt
        self.scroll_content_widget = QWidget()
        self.scroll_content_layout = QVBoxLayout(self.scroll_content_widget)
        self.scroll_content_layout.setContentsMargins(
            40, 20, 40, 40
        )  # Viel "Luft" an den Rändern
        self.scroll_content_layout.setSpacing(25)

        self.right_scroll_area.setWidget(self.scroll_content_widget)
        right_layout.addWidget(self.right_scroll_area)

        # --- INHALT AUFBAUEN ---

        # A. Inputs Container
        self.input_container = QFrame()
        self.input_layout = QVBoxLayout(self.input_container)
        self.input_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_content_layout.addWidget(self.input_container)

        # B. Plot (Breit und Modern)
        self.plot_canvas = PlotCanvas()
        self.plot_canvas.setMinimumHeight(450)
        self.plot_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.scroll_content_layout.addWidget(self.plot_canvas)

        # C. Ergebnisse
        self.results_layout = QVBoxLayout()
        self.results_layout.setSpacing(15)
        self.scroll_content_layout.addLayout(self.results_layout)

        # D. Spacer am Ende
        self.scroll_content_layout.addStretch()

        # --- SPLITTER ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(right_panel)
        self.splitter.setCollapsible(0, True)  # Sidebar darf komplett zugehen
        self.splitter.setSizes([260, 1020])
        self.splitter.setHandleWidth(0)  # Unsichtbarer Griff für cleaner Look

        main_layout.addWidget(self.splitter)

        main_layout.addWidget(self.splitter)

        # Expand first category by default
        top_item = self.sidebar.topLevelItem(0)
        if top_item:
            top_item.setExpanded(True)

    def toggle_sidebar(self):
        """Klappt die Sidebar ein oder aus."""
        if self.sidebar.isVisible():
            self.sidebar.hide()
        else:
            self.sidebar.show()

    def _clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                child_layout = item.layout()
                if child_layout:
                    self._clear_layout(child_layout)

    def on_model_selected(self, item, column):
        # Check if item has model data (UserRole)
        # Categories don't have model data
        model = item.data(0, Qt.ItemDataRole.UserRole)
        
        if model:
            self.current_model = model
            self.setup_input_form()
            self.clear_results()
        else:
            # Clicked on category -> Toggle Expand/Collapse
            if item.isExpanded():
                item.setExpanded(False)
            else:
                item.setExpanded(True)

    def setup_input_form(self):
        self._clear_layout(self.input_layout)
        self.input_widgets = {}

        # Titel groß und modern
        title = QLabel(self.current_model.name)
        title.setObjectName("header")
        self.input_layout.addWidget(title)

        grid = QGridLayout()
        grid.setVerticalSpacing(20)
        grid.setHorizontalSpacing(30)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        variables = self.current_model.get_variables()

        row, col = 0, 0
        for var in variables:
            try:
                v_layout = QVBoxLayout()
                v_layout.setSpacing(6)

                lbl = QLabel(var["label"])
                lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

                widget = None
                if var.get("type") == "dropdown":
                    widget = QComboBox()
                    widget.setView(QListView())
                    widget.setStyleSheet("""
                        QComboBox {
                            background-color: #ffffff;
                            color: #2d3436;
                            border: 1px solid #b2bec3;
                            border-radius: 4px;
                            padding: 5px 10px;
                            min-height: 35px;
                        }
                        QComboBox::drop-down {
                            border: 0px;
                            width: 30px;
                        }
                        QComboBox::down-arrow {
                            image: url(src/assets/chevron_down.svg);
                            width: 12px;
                            height: 12px;
                        }
                        QListView {
                            background-color: #ffffff;
                            color: #2d3436;
                            border: 1px solid #b2bec3;
                            selection-background-color: #3498db;
                            selection-color: #ffffff;
                            outline: none;
                            padding: 5px;
                        }
                    """)
                    for txt, val in var.get("options", []):
                        widget.addItem(txt, val)
                    idx = widget.findData(var["default"])
                    if idx >= 0:
                        widget.setCurrentIndex(idx)

                elif var.get("type") == "text":
                    widget = QLineEdit(str(var["default"]))

                elif isinstance(var["default"], int):
                    widget = QSpinBox()
                    widget.setRange(-999999, 999999)
                    widget.setValue(var["default"])
                    widget.setSingleStep(var.get("step", 1))

                else:
                    widget = QDoubleSpinBox()
                    widget.setRange(-999999.0, 999999.0)
                    widget.setDecimals(4)
                    widget.setValue(var["default"])
                    widget.setSingleStep(var.get("step", 0.1))

                if widget:
                    widget.setMinimumHeight(40)
                    self.input_widgets[var["name"]] = widget

                    v_layout.addWidget(lbl)
                    v_layout.addWidget(widget)

                    grid.addLayout(v_layout, row, col)
                    col += 1
                    if col > 1:
                        col = 0
                        row += 1

            except Exception as e:
                print(f"Error creating input widget for {var.get('name', 'unknown')}: {e}")
                continue

        self.input_layout.addLayout(grid)

        # Berechnen Button (Groß)
        btn_calc = QPushButton("Berechnen")
        btn_calc.setObjectName("btn_calc")  # Spezielles CSS
        btn_calc.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True) # Force Style
        
        # NUCLEAR OPTION: Style direkt im Code erzwingen
        btn_calc.setStyleSheet("""
            QPushButton {
                background-color: #3498db; 
                color: #ffffff;
                border: 1px solid #2980b9;
                border-radius: 6px;
                font-weight: bold;
                font-size: 15px;
                padding: 8px 16px;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #2980b9;
                border: 1px solid #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f618d;
                padding-top: 10px;
            }
        """)
        
        btn_calc.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_calc.clicked.connect(self.run_calculation)

        self.input_layout.addSpacing(30)
        self.input_layout.addWidget(btn_calc)

    def clear_results(self):
        self._clear_layout(self.results_layout)
        self.plot_canvas.clear()

    def run_calculation(self):
        if not self.current_model:
            return
        params = {}
        try:
            for name, widget in self.input_widgets.items():
                if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                    params[name] = widget.value()
                elif isinstance(widget, QComboBox):
                    params[name] = widget.currentData()
                elif isinstance(widget, QLineEdit):
                    params[name] = widget.text()

            result = self.current_model.calculate(**params)
            self.display_results(result)

        except Exception as e:
            QMessageBox.critical(self, "Fehler", str(e))

    def display_results(self, result):
        self.clear_results()

        if result.plot_data:
            self.plot_canvas.plot(result.plot_data)

        for step in result.steps:
            step_widget = QFrame()
            step_widget.setObjectName("step_container")

            l = QVBoxLayout(step_widget)
            l.setContentsMargins(25, 20, 25, 20)
            l.setSpacing(12)

            desc = QLabel(step.description)
            desc.setObjectName("step_desc")
            l.addWidget(desc)

            # Formel immer Schwarz
            latex = LatexLabel(step.latex, font_size=16, color="#000000")
            l.addWidget(latex)

            self.results_layout.addWidget(step_widget)
