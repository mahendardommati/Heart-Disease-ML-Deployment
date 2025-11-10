import flask
from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)

# -------------------------------------------------
# Load all saved models
# -------------------------------------------------

knn_model = pickle.load(open('knn_model.pkl', 'rb'))
logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
naive_model = pickle.load(open('naive_model.pkl', 'rb'))
dt_model = pickle.load(open('dt_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
adaboost_model = pickle.load(open('adaboost_model.pkl', 'rb'))
gb_model = pickle.load(open('gb_model.pkl', 'rb'))
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))


# -------------------------------------------------
# Route for Home Page
# -------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------------------------------
# Route for Prediction
# -------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all form inputs (13 features)
        features = [float(x) for x in request.form.values() if x.replace('.', '', 1).isdigit()]
        model_name = request.form.get("model_name")

        # Convert to NumPy array
        final_features = np.array([features])

        # Select the correct model
        if model_name == "KNN":
            model = knn_model
        elif model_name == "Logistic Regression":
            model = logistic_model
        elif model_name == "Naive Bayes":
            model = naive_model
        elif model_name == "Decision Tree":
            model = dt_model
        elif model_name == "Random Forest":
            model = rf_model
        elif model_name == "AdaBoost":
            model = adaboost_model
        elif model_name == "Gradient Boosting":
            model = gb_model
        elif model_name == "XGBoost":
            model = xgb_model
        elif model_name == "SVM":
            model = svm_model
        else:
            return render_template('index.html', prediction_text="Error: Invalid model selection.")

        # Make prediction
        prediction = model.predict(final_features)[0]
        result = "Heart Disease Detected 💔" if prediction == 1 else "No Heart Disease ❤️"

        return render_template('index.html', prediction_text=f"Result using {model_name}: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")


# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
