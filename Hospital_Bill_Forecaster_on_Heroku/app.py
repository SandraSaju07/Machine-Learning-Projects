# Import required libraries
from flask import Flask, render_template, url_for, request, redirect, jsonify # type: ignore
from pycaret.regression import * # type: ignore
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load transformation pipeline and trained ML model created using PyCaret
model = load_model('./model/gbr_model') # type: ignore

# Define expected input columns
cols = ['age','sex','bmi','children','smoker','region']

# Render home page
@app.route('/')
def home():
    return render_template('home.html')

# Handel predictions
@app.route('/predict', methods=["POST"])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=cols)
    prediction = predict_model(model, data=data_unseen, round=0) # type: ignore
    prediction = (prediction.prediction_label[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

# Run the app
if __name__ == "__main__":
    app.run()