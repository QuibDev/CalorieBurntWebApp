from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('calories_model.sav')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    exercise_type = request.form['exercise_type']
    duration = int(request.form['duration'])
    heart_rate_range = int(request.form['heart_rate_range'])

    # calculate bodyTemp factor 

    skinTempFactor = {'Walking':0.4,'Jogging':1.5,'Running':3.0,'Cycling':2.5,'Squats':1.3,'Push Ups':1.5,'Pull Ups':1.0
                         ,'Arm Curls':0.8,'Lateral Raises':0.8, 'Shoulder Presses':1.3, 'Deadlifts':1.0,'Bench Presses':1.0}

    body_temp = skinTempFactor[exercise_type.capitalize()]

    # Convert gender to binary format
    if gender == 'Male':
        gender = 1
    else:
        gender = 0

    # Make a prediction using the input data and the loaded model
    prediction = np.round(model.predict([[gender, duration, heart_rate_range, body_temp]]), 2)

    # Render the prediction on the webpage
    return render_template('home.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
