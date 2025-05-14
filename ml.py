import sys
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTextEdit, QProgressBar, QMessageBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QThread, Signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import time

class MLWorker(QThread):
    progress_update = Signal(int)
    training_done = Signal(str, float, str, np.ndarray)

    def __init__(self, algo_name, data):
        super().__init__()
        self.algo_name = algo_name
        self.df = data

    def run(self):
        time.sleep(1)
        df = self.df.dropna()
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col])
        X = df.drop("Diagnosis", axis=1)
        y = df["Diagnosis"]
        X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.99, random_state=42)

        if self.algo_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.algo_name == "Logistic Regression":
            model = LogisticRegression(max_iter=500)
        elif self.algo_name == "XGBoost":
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        elif self.algo_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            return

        for i in range(1, 101):
            time.sleep(0.01)
            self.progress_update.emit(i)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        self.training_done.emit(self.algo_name, acc, report, cm)

class MLPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 ML Classifier Dashboard")
        self.setGeometry(100, 100, 1000, 700)
        self.df = None
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #f0f0f0;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #c0c0c0;
                border-radius: 8px;
            }
            QProgressBar {
                background-color: #2e2e2e;
                border: 1px solid #555;
                border-radius: 10px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #00bcd4;
                border-radius: 10px;
            }
        """)

        title = QLabel("🔍 ML Classifier Panel - Thyroid Risk")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        upload_btn = QPushButton("📁 Upload CSV")
        upload_btn.clicked.connect(self.load_data)
        self.style_button(upload_btn)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setFont(QFont("Consolas", 11))

        self.canvas = FigureCanvas(plt.figure(figsize=(6, 4)))
        self.canvas.setStyleSheet("background-color: #1e1e1e;")

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setVisible(False)

        self.buttons_layout = QHBoxLayout()
        self.algo_buttons = {}
        for name in ["Random Forest", "Logistic Regression", "XGBoost", "KNN"]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, n=name: self.run_model(n))
            self.style_button(btn)
            btn.setEnabled(False)
            self.algo_buttons[name] = btn
            self.buttons_layout.addWidget(btn)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(upload_btn)
        layout.addLayout(self.buttons_layout)
        layout.addWidget(self.progress)
        layout.addWidget(self.result_box)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def style_button(self, btn):
        btn.setStyleSheet("""
            QPushButton {
                background-color: #03a9f4;
                color: white;
                padding: 12px;
                border-radius: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #0288d1;
            }
        """)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.df = pd.read_csv(path)
                self.result_box.append("✅ CSV Loaded Successfully.")
                for btn in self.algo_buttons.values():
                    btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def run_model(self, algo_name):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please upload a dataset first.")
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.result_box.clear()
        self.worker = MLWorker(algo_name, self.df)
        self.worker.progress_update.connect(self.progress.setValue)
        self.worker.training_done.connect(self.display_results)
        self.worker.start()

    def display_results(self, algo_name, accuracy, report, cm):
        self.result_box.append(f"📌 Results for {algo_name}")
        self.result_box.append(f"✅ Accuracy: {accuracy:.2f}%\n")
        self.result_box.append(report)

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt="d", cmap="rocket", ax=ax)
        ax.set_title(f"{algo_name} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        self.canvas.draw()

        self.progress.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLPage()
    window.show()
    sys.exit(app.exec())