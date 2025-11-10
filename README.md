🩺 Heart Disease Prediction - Machine Learning Project
📌 Project Overview

This is an end-to-end Machine Learning project designed to predict whether a person is likely to have heart disease based on various health parameters.
The project includes data preprocessing, model training, performance evaluation, and deployment using Flask.
A clean and functional front-end interface allows users to enter values and get instant predictions.

🧠 Models Used

This project includes training and comparison of 8 major classification algorithms:

K-Nearest Neighbors (KNN)

Logistic Regression

Naive Bayes

Decision Tree

Random Forest

AdaBoost

Gradient Boosting

XGBoost

Support Vector Machine (SVM)

All models were trained using the Heart Disease dataset from Kaggle.

Dataset Link: Heart Disease Dataset - Kaggle

⚙️ Tech Stack

Programming Language: Python

Web Framework: Flask

Libraries Used:
numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, flask, pickle

Front-End: HTML, CSS

Deployment: Local Flask Server

💾 Project Structure
Mini_Project_ML/
│
├── app.py                      # Flask Application
├── templates/
│   └── index.html              # Front-end HTML page
├── models/
│   ├── knn_model.pkl
│   ├── logistic_model.pkl
│   ├── naive_model.pkl
│   ├── dt_model.pkl
│   ├── rf_model.pkl
│   ├── adaboost_model.pkl
│   ├── gb_model.pkl
│   ├── xgb_model.pkl
│   └── svm_model.pkl
├── heart.csv                   # Dataset
└── README.md                   # Project Description

🚀 How to Run the Project
Step 1️⃣ - Clone the Repository
git clone https://github.com/YourUsername/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

Step 2️⃣ - Install Dependencies
pip install -r requirements.txt

Step 3️⃣ - Run the Flask App
python app.py

Step 4️⃣ - Open in Browser

Go to:
👉 http://127.0.0.1:5000
