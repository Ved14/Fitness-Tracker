import os
from flask import Flask, request, render_template
import pandas as pd
import joblib
from preprocess_and_predict import main

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_exercise', methods=['POST'])
def predict_exercise():
    if 'acc_file' not in request.files or 'gyr_file' not in request.files:
        return "No file part"
    acc_file = request.files['acc_file']
    gyr_file = request.files['gyr_file']
    if acc_file.filename == '' or gyr_file.filename == '':
        return "No selected file"

    acc_file_path = os.path.join('uploads', acc_file.filename)
    gyr_file_path = os.path.join('uploads', gyr_file.filename)
    acc_file.save(acc_file_path)
    gyr_file.save(gyr_file_path)
    
    predictions, prediction_probabilities, accuracy, reps_count, plot_filename = main(acc_file_path, gyr_file_path)
    
    exercise_names = {
            'bench': 'Bench Press',
            'squat': 'Squat',
            'ohp': 'Over Head Press',
            'dead': 'Deadlift',
            'row': 'Row',
            'rest': 'Rest'
        }

        # Transform predictions using the mapping
    predicted_exercise = exercise_names.get(predictions[0], predictions[0])

    return render_template('results.html', 
                           exercise=predicted_exercise, 
                           probability=round(prediction_probabilities[0].max() * 100,2), 
                           accuracy=accuracy*100, 
                           reps_count=reps_count,
                           plot_filename=plot_filename)

if __name__ == '__main__':
    app.run(debug=True)
