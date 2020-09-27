import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('DiabetesPredictor.pkl', 'rb'))
scaler = pickle.load(open('StandardScaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    pregnancies = int(float(request.form['pregnancies']))
    glucLevel = int(float(request.form['glucLevel']))
    bloodPres = int(float(request.form['bloodPres']))
    skinThick = int(float(request.form['skinThick']))
    insulinLevel = int(float(request.form['insulinLevel']))
    bmi = float(request.form['bmi'])
    pedigreeFunc = float(request.form['pedigreeFunc'])
    age = int(float(request.form['age']))

    scaledData = scaler.transform(np.array([[pregnancies, glucLevel, bloodPres, skinThick, insulinLevel, bmi, pedigreeFunc, age]]))

    prediction = model.predict(scaledData)

    if prediction == 1:
        return render_template('index.html', prediction_text='You are likely to have diabetes.')
    else:
        return render_template('index.html', prediction_text='You are safe.')


if __name__ == "__main__":
    app.run(debug=True)
