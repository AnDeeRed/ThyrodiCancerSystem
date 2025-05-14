import sys
import subprocess
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

class ButtonsPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Thyroid Risk Prediction System")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("background-color: #121826; color: white;")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignCenter)

        # Title
        title = QLabel("🧠 Thyroid Risk Prediction System")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00d4ff;")
        layout.addWidget(title)

        # Subtitle/Description
        desc = QLabel(
            "🔍 This system empowers medical researchers and practitioners to:\n"
            "• Predict thyroid cancer risk using Machine Learning & Neural Networks\n"
            "• Analyze hormone & demographic patterns via Visualizations\n"
            "• Explore clustering & pattern mining in patients\n"
            "• Retrieve and interpret medical insights from datasets"
        )
        desc.setFont(QFont("Segoe UI", 13))
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Spacer
        layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # Button Layout
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(30)
        btn_layout.setAlignment(Qt.AlignCenter)

        buttons = [
            ("🧠 Neural Network", "nn.py"),
            ("📊 Machine Learning", "ml.py"),
            ("📁 Information Retrieval", "selection.py"),
            ("📉 Clustering", "clustering.py"),
            ("📈 Visualizations", "visualisations.py"),
            ("🧾 Prediction", "classifier.py"),
        ]

        for label, script in buttons:
            btn = QPushButton(label)
            btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
            btn.setFixedSize(230, 60)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #00c2ff;
                    border-radius: 12px;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #0099cc;
                }
            """)
            btn.clicked.connect(lambda checked, s=script: self.launch_script(s))
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)

        # Set layout
        self.setLayout(layout)

    def launch_script(self, script_name):
        try:
            subprocess.Popen(["python", script_name])
        except Exception as e:
            print(f"❌ Failed to launch {script_name}: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ButtonsPage()
    window.show()
    sys.exit(app.exec())
