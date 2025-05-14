import sys, threading, time, numpy as np, pandas as pd
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QFileDialog, QTextEdit,
                               QVBoxLayout, QHBoxLayout, QProgressBar, QComboBox, QSpinBox)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QThread, Signal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN, LSTM
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import seaborn as sns

class TrainerThread(QThread):
    progress_update = Signal(int, str)
    training_done = Signal(float, str, object)

    def __init__(self, df, model_name, batch, epochs):
        super().__init__()
        self.df = df
        self.model_name = model_name
        self.batch = batch
        self.epochs = epochs
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self):
        df = self.df.dropna()
        le = LabelEncoder()
        for col in df.select_dtypes(include="object"):
            df[col] = le.fit_transform(df[col])

        X = df.drop("Diagnosis", axis=1)
        y = le.fit_transform(df["Diagnosis"])
        y_cat = to_categorical(y)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.99)

        input_shape = (X_train.shape[1], 1)
        model = Sequential()

        if self.model_name == "DNN":
            model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(y_cat.shape[1], activation='softmax'))
        else:
            X_train = X_train.reshape(-1, X_train.shape[1], 1)
            X_test = X_test.reshape(-1, X_test.shape[1], 1)
            if self.model_name == "CNN":
                model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
                model.add(MaxPooling1D(2))
                model.add(Flatten())
            elif self.model_name == "RNN":
                model.add(SimpleRNN(64, input_shape=input_shape))
            elif self.model_name == "LSTM":
                model.add(LSTM(64, input_shape=input_shape))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(y_cat.shape[1], activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        start = time.time()
        for i in range(1, self.epochs + 1):
            if self.stop_flag:
                return
            model.fit(X_train, y_train, epochs=1, batch_size=self.batch, verbose=0)
            elapsed = time.time() - start
            eta = (elapsed / i) * (self.epochs - i)
            self.progress_update.emit(int((i / self.epochs) * 100), f"⏳ ETA: {int(eta)}s (Epoch {i}/{self.epochs})")

        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        acc = accuracy_score(y_true, y_pred_labels) * 100
        report = classification_report(y_true, y_pred_labels)

        cm = confusion_matrix(y_true, y_pred_labels)
        self.training_done.emit(acc, report, cm)

class DeepLearningApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Deep Learning Trainer")
        self.setGeometry(200, 100, 1000, 700)
        self.setStyleSheet("background-color: white; color: #111;")
        self.thread = None
        self.df = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("📊 Deep Learning Risk Trainer")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        upload_btn = QPushButton("📂 Upload CSV")
        upload_btn.setStyleSheet("background-color: #03a9f4; color: white; padding: 10px;")
        upload_btn.clicked.connect(self.upload_csv)
        layout.addWidget(upload_btn)

        config = QHBoxLayout()
        self.epoch_box = QSpinBox(); self.epoch_box.setValue(20); self.epoch_box.setPrefix("Epochs: "); config.addWidget(self.epoch_box)
        self.batch_box = QSpinBox(); self.batch_box.setValue(32); self.batch_box.setPrefix("Batch: "); config.addWidget(self.batch_box)
        self.model_box = QComboBox(); self.model_box.addItems(["DNN", "CNN", "RNN", "LSTM"]); config.addWidget(self.model_box)
        layout.addLayout(config)

        button_row = QHBoxLayout()
        self.train_btn = QPushButton("🚀 Train Model")
        self.train_btn.setStyleSheet("background-color: #4caf50; color: white; padding: 10px; font-weight: bold;")
        self.train_btn.clicked.connect(self.train_model)
        button_row.addWidget(self.train_btn)

        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_training)
        button_row.addWidget(self.cancel_btn)

        layout.addLayout(button_row)

        self.eta_label = QLabel("⏳ ETA: -")
        layout.addWidget(self.eta_label)

        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        layout.addWidget(self.progress)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        self.setLayout(layout)

    def upload_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path)
            self.output.append("✅ Dataset loaded successfully.")

    def cancel_training(self):
        if self.thread:
            self.thread.stop()
            self.output.append("⚠️ Training canceled.")
            self.eta_label.setText("❌ Cancelled")
            self.progress.setValue(0)
            self.train_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def train_model(self):
        if self.df is None:
            self.output.append("❗ Please upload a dataset first.")
            return

        self.train_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        model = self.model_box.currentText()
        batch = self.batch_box.value()
        epochs = self.epoch_box.value()

        self.thread = TrainerThread(self.df.copy(), model, batch, epochs)
        self.thread.progress_update.connect(self.update_progress)
        self.thread.training_done.connect(self.training_finished)
        self.thread.start()

    def update_progress(self, value, eta):
        self.progress.setValue(value)
        self.eta_label.setText(eta)

    def training_finished(self, acc, report, cm):
        self.output.append(f"\n✅ Accuracy: {acc:.2f}%\n\n{report}")
        self.progress.setValue(100)
        self.eta_label.setText("✅ Training Completed")
        self.train_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        # Plotting results (console popup)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
        ax[0].set_title("Confusion Matrix")

        ax[1].bar(["Accuracy"], [acc], color='green')
        ax[1].set_ylim(0, 100)
        ax[1].set_title("Accuracy Bar")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepLearningApp()
    window.show()
    sys.exit(app.exec())
