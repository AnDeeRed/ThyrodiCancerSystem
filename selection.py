import sys
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QTableWidget,
    QTableWidgetItem, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt
import os

class InfoRetrievalDark(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Information Retrieval - Thyroid Risk Dataset")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("background-color: #121212; color: #e0e0e0;")
        self.data = pd.DataFrame()
        self.filters = {}
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout()

        title = QLabel("🔎 Retrieve Patient Records")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #00d4ff; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        load_btn = QPushButton("📂 Load Dataset")
        load_btn.setStyleSheet("background-color: #007acc; color: white; font-size: 16px; padding: 10px;")
        load_btn.clicked.connect(self.load_data)
        layout.addWidget(load_btn)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_area = QWidget()
        self.scroll_layout = QHBoxLayout(self.scroll_area)
        self.left_filter = QVBoxLayout()
        self.right_filter = QVBoxLayout()
        self.scroll_layout.addLayout(self.left_filter)
        self.scroll_layout.addLayout(self.right_filter)
        self.scroll.setWidget(self.scroll_area)
        layout.addWidget(self.scroll)

        self.filter_btn = QPushButton("✅ Apply Filters")
        self.filter_btn.setStyleSheet("background-color: #27ae60; color: white; font-size: 16px; padding: 10px;")
        self.filter_btn.setEnabled(False)
        self.filter_btn.clicked.connect(self.apply_filters)
        layout.addWidget(self.filter_btn)

        self.count_label = QLabel("Patients Retrieved: 0")
        self.count_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 10px; color: #00ffcc;")
        layout.addWidget(self.count_label)

        self.table = QTableWidget()
        self.table.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        layout.addWidget(self.table)

        self.setLayout(layout)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open file", "", "CSV Files (*.csv *.xlsx)")
        if path:
            ext = os.path.splitext(path)[1]
            try:
                self.data = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)
                self.setup_filters()
                QMessageBox.information(self, "Loaded", "✅ Dataset loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def setup_filters(self):
        for layout in [self.left_filter, self.right_filter]:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

        self.filters = {}
        mid = len(self.data.columns) // 2

        for i, col in enumerate(self.data.columns):
            layout = self.left_filter if i < mid else self.right_filter
            hbox = QHBoxLayout()
            label = QLabel(col)
            label.setStyleSheet("color: #ffa500;")
            label.setFixedWidth(130)
            hbox.addWidget(label)

            if pd.api.types.is_numeric_dtype(self.data[col]):
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                exact = QLineEdit(); exact.setPlaceholderText("Exact"); exact.setStyleSheet("background-color: #2e2e2e; color: #ffffff;")
                minval = QLineEdit(); minval.setPlaceholderText(f"Min: {min_val}"); minval.setStyleSheet("background-color: #2e2e2e; color: #ffffff;")
                maxval = QLineEdit(); maxval.setPlaceholderText(f"Max: {max_val}"); maxval.setStyleSheet("background-color: #2e2e2e; color: #ffffff;")
                exact.setFixedWidth(70); minval.setFixedWidth(70); maxval.setFixedWidth(70)
                hbox.addWidget(exact); hbox.addWidget(minval); hbox.addWidget(maxval)
                self.filters[col] = ("num", exact, minval, maxval)
            else:
                combo = QComboBox()
                combo.setStyleSheet("background-color: #2e2e2e; color: #ffffff;")
                combo.addItem("")
                combo.addItems(sorted(self.data[col].astype(str).dropna().unique()))
                combo.setFixedWidth(180)
                hbox.addWidget(combo)
                self.filters[col] = ("cat", combo)

            layout.addLayout(hbox)

        self.filter_btn.setEnabled(True)

    def apply_filters(self):
        df = self.data.copy()
        for col, config in self.filters.items():
            if config[0] == "num":
                exact, minval, maxval = config[1].text(), config[2].text(), config[3].text()
                try:
                    if exact: df = df[df[col] == float(exact)]
                    if minval: df = df[df[col] >= float(minval)]
                    if maxval: df = df[df[col] <= float(maxval)]
                except ValueError:
                    continue  # Skips filtering when conversion to float fails
            else:
                selected = config[1].currentText()
                if selected: df = df[df[col].astype(str).str.lower() == selected.lower()]

        self.update_table(df)

    def update_table(self, df):
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)

        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))

        self.count_label.setText(f"Patients Retrieved: {len(df)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InfoRetrievalDark()
    window.show()
    sys.exit(app.exec())
