import sys
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog,
    QTextEdit, QHBoxLayout, QFrame, QScrollArea, QMessageBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from kmodes.kprototypes import KPrototypes


class ClusteringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Clustering Analysis")
        self.setMinimumSize(1200, 700)
        self.setStyleSheet("background-color: #121212; color: white;")
        self.df = None
        self.canvas = None
        self.explanation = None
        self.init_ui()

    def init_ui(self):
        title = QLabel("🧠 Clustering for Thyroid Risk Data")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        upload_btn = QPushButton("📂 Upload CSV")
        upload_btn.setFont(QFont("Segoe UI", 14))
        upload_btn.clicked.connect(self.upload_csv)
        upload_btn.setStyleSheet("background-color: #1f77b4; color: white;")

        self.kmeans_btn = QPushButton("📊 K-Means")
        self.kmeans_btn.clicked.connect(lambda: self.run_clustering("kmeans"))
        self.kmeans_btn.setEnabled(False)

        self.dbscan_btn = QPushButton("🌐 DBSCAN")
        self.dbscan_btn.clicked.connect(lambda: self.run_clustering("dbscan"))
        self.dbscan_btn.setEnabled(False)

        self.kproto_btn = QPushButton("🔣 K-Prototypes")
        self.kproto_btn.clicked.connect(lambda: self.run_clustering("kprototypes"))
        self.kproto_btn.setEnabled(False)

        for btn in [self.kmeans_btn, self.dbscan_btn, self.kproto_btn]:
            btn.setFont(QFont("Segoe UI", 14))
            btn.setStyleSheet("background-color: #ff7f0e; color: white; padding: 10px;")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.kmeans_btn)
        btn_layout.addWidget(self.dbscan_btn)
        btn_layout.addWidget(self.kproto_btn)

        self.plot_frame = QFrame()
        self.plot_layout = QVBoxLayout()
        self.plot_frame.setLayout(self.plot_layout)

        self.explanation = QTextEdit()
        self.explanation.setReadOnly(True)
        self.explanation.setStyleSheet("background-color: #1e1e1e; color: white; padding: 10px; font-size: 13px;")
        self.explanation.setFont(QFont("Segoe UI", 11))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.explanation)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(upload_btn)
        layout.addLayout(btn_layout)
        layout.addWidget(self.plot_frame)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def upload_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path)
            self.df.columns = [col.strip() for col in self.df.columns]  # Remove leading/trailing spaces
            self.kmeans_btn.setEnabled(True)
            self.dbscan_btn.setEnabled(True)
            self.kproto_btn.setEnabled(True)

    def find_columns(self, required_names):
        found = []
        for name in required_names:
            for col in self.df.columns:
                if name.lower() in col.lower():
                    found.append(col)
                    break
        return found if len(found) == len(required_names) else None

    def run_clustering(self, algo):
        self.clear_plots()
        if self.df is None:
            return

        if algo == "kmeans":
            features = self.find_columns(['TSH', 'T3', 'T4'])
            if not features:
                return self.show_error("TSH, T3, or T4 columns not found in the dataset.")
            data = self.df[features].dropna()
            scaled = StandardScaler().fit_transform(data)
            model = KMeans(n_clusters=3, random_state=42)
            labels = model.fit_predict(scaled)
            self.show_plot(data, labels, "K-Means Clustering", features[1], features[2])
            self.explanation.setPlainText("✅ K-Means used features: " + ", ".join(features))

        elif algo == "dbscan":
            features = self.find_columns(['TSH', 'T3'])
            if not features:
                return self.show_error("TSH or T3 columns not found in the dataset.")
            data = self.df[features].dropna()
            scaled = StandardScaler().fit_transform(data)
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(scaled)
            self.show_plot(data, labels, "DBSCAN Clustering", features[0], features[1])
            self.explanation.setPlainText("✅ DBSCAN used features: " + ", ".join(features))

        elif algo == "kprototypes":
            features = self.find_columns(['Gender', 'Blood_Pressure', 'Diabetes'])
            if not features:
                return self.show_error("Gender, Blood Pressure, or Diabetes columns not found.")
            data = self.df[features].dropna().copy()
            enc = LabelEncoder()
            for col in features:
                data[col] = enc.fit_transform(data[col])
            model = KPrototypes(n_clusters=3, init='Cao', n_init=5, verbose=0)
            labels = model.fit_predict(data.to_numpy(), categorical=[0, 1, 2])
            self.show_plot(data, labels, "K-Prototypes Clustering", features[0], features[1])
            self.explanation.setPlainText("✅ K-Prototypes used features: " + ", ".join(features))

    def show_plot(self, data, labels, title, x_label, y_label):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(data[x_label], data[y_label], c=labels, cmap="Set1")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)

        canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(canvas)
        self.canvas = canvas

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

    def clear_plots(self):
        if self.canvas:
            self.canvas.setParent(None)
            self.canvas = None
        self.explanation.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClusteringApp()
    window.show()
    sys.exit(app.exec())
