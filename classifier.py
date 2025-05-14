import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QFormLayout, QPushButton,
    QFileDialog, QLineEdit, QComboBox, QMessageBox, QFrame
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

class ClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧬 Thyroid Cancer Risk Predictor")
        self.setMinimumSize(1080, 860)
        self.setStyleSheet("background-color: #0b0f1a; color: #e0e0e0; font-family: Segoe UI;")
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.features = []
        self.inputs = {}
        self.df = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("🧬 Predict Thyroid Cancer Risk")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00e6e6; margin-top: 20px;")
        layout.addWidget(title)

        upload_btn = QPushButton("📁 Upload Dataset")
        upload_btn.clicked.connect(self.upload_dataset)
        upload_btn.setStyleSheet(self.button_style("#2980b9"))
        upload_btn.setFixedHeight(50)
        layout.addWidget(upload_btn, alignment=Qt.AlignCenter)

        self.status_label = QLabel("📄 Waiting for dataset upload...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; color: #cccccc; margin-bottom: 10px;")
        layout.addWidget(self.status_label)

        self.form_layout = QFormLayout()
        self.form_frame = QFrame()
        self.form_frame.setStyleSheet("background-color: #111827; border-radius: 12px; padding: 20px;")
        self.form_frame.setLayout(self.form_layout)
        layout.addWidget(self.form_frame)

        self.predict_btn = QPushButton("🔮 Predict Risk")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setStyleSheet(self.button_style("#27ae60"))
        self.predict_btn.setFixedHeight(50)
        layout.addWidget(self.predict_btn, alignment=Qt.AlignCenter)

        self.result_frame = QFrame()
        self.result_layout = QVBoxLayout()
        self.result_frame.setLayout(self.result_layout)
        self.result_frame.setStyleSheet("background-color: #1e293b; border-radius: 12px; padding: 20px;")
        layout.addWidget(self.result_frame)

        self.setLayout(layout)

    def button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 12px;
                font-size: 16px;
                border-radius: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #1abc9c;
            }}
        """

    def upload_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path).dropna()
            target = "Thyroid_Cancer_Risk"
            if target not in df.columns:
                raise ValueError("Missing required column: Thyroid_Cancer_Risk")

            self.original_df = df.copy()
            self.label_encoders = {}

            for col in df.select_dtypes(include="object").columns:
                if col != target:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col.lower()] = le

            self.target_encoder = LabelEncoder()
            df[target] = self.target_encoder.fit_transform(df[target])

            self.features = [col for col in df.columns if col not in [target, "Patient_ID", "Diagnosis"]]
            X = df[self.features]
            y = df[target]

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.model = LogisticRegression(max_iter=500)
            self.model.fit(X_scaled, y)

            self.df = pd.read_csv(file_path)
            self.status_label.setText("✅ Model trained. Enter values to predict risk.")
            self.build_input_fields()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def build_input_fields(self):
        for i in reversed(range(self.form_layout.count())):
            self.form_layout.itemAt(i).widget().setParent(None)
        self.inputs.clear()

        for col in self.features:
            label = QLabel(f"{col}")
            label.setStyleSheet("color: #00e6e6; font-size: 14px; padding: 4px;")

            if self.df[col].dtype == object or self.df[col].nunique() <= 15:
                combo = QComboBox()
                combo.setStyleSheet("background-color: #1e293b; color: white; padding: 6px;")
                combo.addItem("")
                values = sorted(self.df[col].dropna().unique().tolist())
                combo.addItems([str(v) for v in values])
                self.inputs[col] = combo
                self.form_layout.addRow(label, combo)
            else:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                exact_field = QLineEdit()
                exact_field.setPlaceholderText(f"Enter {col} ({min_val} - {max_val})")
                exact_field.setStyleSheet("background-color: #0f172a; color: white; padding: 8px; border-radius: 6px;")
                self.inputs[col] = exact_field
                self.form_layout.addRow(label, exact_field)

        self.predict_btn.setEnabled(True)

    def predict(self):
        try:
            values = []
            for col in self.features:
                widget = self.inputs[col]
                if isinstance(widget, QComboBox):
                    text = widget.currentText().strip()
                    if not text:
                        raise ValueError(f"{col} cannot be empty.")
                    le = self.label_encoders.get(col.lower())
                    val = le.transform([text])[0] if le else int(text)
                else:
                    val = float(widget.text().strip())
                values.append(val)

            X_scaled = self.scaler.transform([values])
            pred = self.model.predict(X_scaled)[0]
            label = self.target_encoder.inverse_transform([pred])[0]
            self.show_result(label)
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))

    def show_result(self, label):
        for i in reversed(range(self.result_layout.count())):
            self.result_layout.itemAt(i).widget().setParent(None)

        result = QLabel(f"🧾 Predicted Risk: <b>{label}</b>")
        result.setAlignment(Qt.AlignCenter)
        result.setStyleSheet("font-size: 18px; color: #2ecc71; font-weight: bold;")
        self.result_layout.addWidget(result)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClassifierApp()
    win.show()
    sys.exit(app.exec())
