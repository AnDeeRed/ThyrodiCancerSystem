Overview
This project focuses on predicting the likelihood of diabetes in patients using their vital signs and medical history, leveraging various machine learning algorithms. It is structured to compare model performance and identify the most reliable algorithm for clinical implementation, with a focus on interpretability and predictive accuracy.

Key Features

Dataset: A structured dataset with 100,000 records and 8 key features such as BMI, glucose levels, hypertension, age, and smoking history.

Algorithms Used:

Random Forest: Achieved the highest performance with 97.03% accuracy and 96.97% precision.

K-Nearest Neighbors (KNN): Delivered solid predictive accuracy (95.24%) with strong interpretability.

Naïve Bayes: Noted for fast computation and robustness with categorical data.

Preprocessing: Involves label encoding, Min-Max scaling, and correlation analysis to prepare clean, normalized input for model training.

Validation: Applies 10-fold cross-validation for all models to ensure generalization and mitigate overfitting.

Performance Metrics: Includes accuracy, precision, F1-score, and ROC-AUC to evaluate model efficacy.

Goal
This project demonstrates how data-driven methods can assist in early diabetes detection by analyzing key clinical features, thereby enabling proactive healthcare interventions and more efficient patient monitoring.
