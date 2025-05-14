import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QScrollArea, QFrame
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

class VisualizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.setWindowTitle("📊 Thyroid Risk Visualizations")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #121212; color: white;")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("📊 Thyroid Risk Prediction System Visualizations")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00e6e6; margin: 20px;")
        layout.addWidget(title)

        upload_btn = QPushButton("📁 Upload Dataset")
        upload_btn.setFont(QFont("Segoe UI", 14))
        upload_btn.setStyleSheet("background-color: #34495e; color: white; font-size: 16px; padding: 10px; border-radius: 10px;")
        upload_btn.clicked.connect(self.load_data)
        layout.addWidget(upload_btn, alignment=Qt.AlignCenter)

        # Scroll area to contain buttons
        self.scroll_area = QScrollArea()
        self.scroll_area.setStyleSheet("background-color: #121212; border: none;")
        self.scroll_area.setWidgetResizable(True)
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)  # Use QVBoxLayout for a single side of buttons
        button_container.setLayout(button_layout)

        # List of visualization functions and their names with emojis
        visualizations = [
            ("👶 Risk by Age", self.plot_age_distribution),
            ("🚻 Gender-wise Risk", self.plot_gender_risk),
            ("🌍 Country-wise Count", self.plot_country_distribution),
            ("🏋️‍♂️ Risk vs Obesity/Diabetes", self.plot_obesity_diabetes),
            ("👨‍👩‍👧‍👦 Family History vs Risk", self.plot_family_history_risk),
            ("☢️ Radiation Risk", self.plot_radiation_risk),
            ("📈 Feature Importance", self.plot_feature_importance)
        ]

        # Colors for buttons
        colors = ['#16a085', '#c0392b', '#2980b9', '#8e44ad', '#d35400', '#f39c12', '#27ae60']
        for (name, func), color in zip(visualizations, colors):
            btn = QPushButton(name)
            btn.setFont(QFont("Segoe UI", 12))
            btn.setMinimumHeight(40)
            btn.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; border-radius: 8px; padding: 8px; }}"
                              f"QPushButton:hover {{ background-color: #333; }}")
            btn.clicked.connect(func)
            button_layout.addWidget(btn)

        self.scroll_area.setWidget(button_container)
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path)
            self.df.columns = self.df.columns.str.strip()
            for col in self.df.select_dtypes(include=['object']).columns:
                self.df[col] = self.df[col].astype('category')

    def plot_age_distribution(self):
        plt.figure()
        plt.hist(self.df['Age'], bins=20, color='skyblue', edgecolor='black')
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_gender_risk(self):
        if 'Gender' in self.df.columns and 'Thyroid_Cancer_Risk' in self.df.columns:
            plt.figure()
            counts = self.df.groupby('Gender')['Thyroid_Cancer_Risk'].value_counts().unstack().plot(kind='bar', colormap='viridis')
            plt.title("Gender-wise Risk Distribution")
            plt.xlabel("Gender")
            plt.ylabel("Risk Level Count")
            plt.legend(title="Risk Level")
            plt.tight_layout()
            plt.show()

    def plot_country_distribution(self):
        plt.figure()
        counts = self.df['Country'].value_counts()
        plt.barh(counts.index, counts.values, color=cm.tab20.colors)
        plt.title("Patient Distribution by Country")
        plt.xlabel("Number of Patients")
        plt.tight_layout()
        plt.show()

    def plot_obesity_diabetes(self):
        plt.figure()
        counts = self.df.groupby(['Obesity', 'Diabetes'])['Thyroid_Cancer_Risk'].count().unstack()
        counts.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title("Obesity and Diabetes vs Risk")
        plt.ylabel("Patient Count")
        plt.tight_layout()
        plt.show()

    def plot_family_history_risk(self):
        plt.figure()
        counts = self.df.groupby(['Family_History', 'Thyroid_Cancer_Risk']).size().unstack()
        counts.plot(kind='bar', stacked=True, colormap='coolwarm')
        plt.title("Family History vs Risk")
        plt.tight_layout()
        plt.show()

    def plot_radiation_risk(self):
        plt.figure()
        counts = self.df.groupby(['Radiation_Exposure', 'Thyroid_Cancer_Risk']).size().unstack()
        counts.plot(kind='bar', stacked=True, colormap='autumn')
        plt.title("Radiation Exposure vs Risk")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        plt.figure()
        df = self.df.dropna()
        X = df.drop('Thyroid_Cancer_Risk', axis=1)
        y = LabelEncoder().fit_transform(df['Thyroid_Cancer_Risk'])
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=True)
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance.nlargest(10).plot(kind='barh', color=cm.viridis.colors)
        plt.title("Feature Importance (XGBoost)")
        plt.tight_layout()
        plt.show()

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VisualizationApp()
    win.show()
    sys.exit(app.exec())
