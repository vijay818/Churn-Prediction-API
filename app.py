# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 12:44:35 2020

@author: SPIN_Data Science
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
app.url_map.strict_slashes = False
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    '''
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('form.html', prediction_text = 'Churn value is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)