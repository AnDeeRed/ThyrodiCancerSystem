import sys
import subprocess
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

class MainPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Thyroid Risk Prediction System")
        self.setMinimumSize(900, 600)
        self.setStyleSheet("background-color: #1e1e2f; color: white;")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("🧠 Thyroid Risk Prediction System")
        title.setFont(QFont("Segoe UI", 30, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00c2ff; margin-bottom: 15px;")

        subtitle = QLabel("🧪 CS 594 – DSA-CD | Spring 2025")
        subtitle.setFont(QFont("Segoe UI", 20))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #ffcc00; margin-bottom: 30px;")

        members = QLabel(
            "👨‍🎓 Rishi Sai Subhash Pasupuleti- 362712\n"
            "👩‍🎓 Srivalli Patibandla - 365309\n"
            "👨‍🎓 Misbah Mohammed - 362841\n"
            "👩‍🎓 Sravya Kunapareddy - 365093\n"
            "👨‍🎓 Deepak Reddy Anumula - 360817"
        )
        members.setFont(QFont("Segoe UI", 14))
        members.setAlignment(Qt.AlignCenter)
        members.setStyleSheet("margin-bottom: 40px;")

        start_btn = QPushButton("🚀 Start Project")
        start_btn.setFont(QFont("Segoe UI", 16, QFont.Bold))
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #00c2ff;
                color: white;
                padding: 15px 30px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #009fc2;
            }
        """)
        start_btn.clicked.connect(self.launch_buttons_page)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(members)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def launch_buttons_page(self):
        subprocess.Popen(["python", "buttons.py"])
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainPage()
    window.show()
    sys.exit(app.exec())
