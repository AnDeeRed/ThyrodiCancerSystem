import sys
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTextEdit, QProgressBar, QMessageBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QThread, Signal, QObject
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering, Birch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ClusteringWorker(QObject):
    finished = Signal(str, np.ndarray, np.ndarray, list)
    error = Signal(str)

    def __init__(self, algo_name, data):
        super().__init__()
        self.algo_name = algo_name
        self.df = data

    def run(self):
        try:
            df = self.df.copy().dropna()
            le = LabelEncoder()
            categorical_cols = df.select_dtypes(include='object').columns.tolist()
            for col in categorical_cols:
                df[col] = le.fit_transform(df[col].astype(str))

            important = ['age', 'tsh', 't3', 't4', 'thyroid_cancer_risk', 'obesity', 'family_history']
            important_features = [col for col in df.columns if col.lower() in important]
            X = df[important_features]
            selected_features = important_features[:2] if len(important_features) >= 2 else X.columns.tolist()[:2]
            data_2d = X[selected_features].values

            if self.algo_name == "Agglomerative Clustering":
                model = AgglomerativeClustering(n_clusters=3)
                labels = model.fit_predict(X)
            elif self.algo_name == "Birch":
                model = Birch(n_clusters=3)
                labels = model.fit_predict(X)
            else:
                raise ValueError("Unsupported algorithm")

            self.finished.emit(self.algo_name, labels, data_2d, selected_features)
        except Exception as e:
            self.error.emit(str(e))

class ClusteringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📉 Clustering Analysis - Thyroid Data")
        self.setGeometry(150, 150, 1000, 700)
        self.df = None
        self.thread = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QWidget { background-color: #121212; color: #f0f0f0; }
            QTextEdit { background-color: #1e1e1e; color: #c0c0c0; border-radius: 8px; }
            QProgressBar {
                background-color: #2e2e2e; border: 1px solid #555; border-radius: 10px;
                text-align: center; color: #ffffff; }
            QProgressBar::chunk { background-color: #00bcd4; border-radius: 10px; }
        """)

        title = QLabel("📉 Clustering Module - Thyroid Risk")
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
        self.progress.setVisible(False)

        self.buttons_layout = QHBoxLayout()
        self.algo_buttons = {}
        for name in ["Agglomerative Clustering", "Birch"]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, n=name: self.run_clustering(n))
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
                background-color: #03a9f4; color: white; padding: 12px;
                border-radius: 10px; font-size: 16px;
            }
            QPushButton:hover { background-color: #0288d1; }
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

    def run_clustering(self, algo_name):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please upload a dataset first.")
            return

        self.progress.setVisible(True)
        self.progress.setValue(50)
        self.result_box.clear()

        self.worker = ClusteringWorker(algo_name, self.df)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def display_results(self, algo_name, labels, data, features):
        self.result_box.append(f"📌 Results for {algo_name}")
        self.result_box.append(f"✅ Total Clusters: {len(np.unique(labels))}")
        self.result_box.append(f"📈 Features used in plot: {features[0]} (X-axis), {features[1]} (Y-axis)")

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=40)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title(f"{algo_name} - Cluster Plot")
        ax.legend(*scatter.legend_elements(), title="Clusters")
        self.canvas.draw()

        self.progress.setVisible(False)

    def show_error(self, message):
        QMessageBox.critical(self, "Clustering Error", message)
        self.progress.setVisible(False)

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClusteringApp()
    window.show()
    sys.exit(app.exec())
