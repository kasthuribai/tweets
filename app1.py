# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:18:53 2021

@author: USER
"""

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from sklearn import linear_model
#from sklearn.externals import joblib
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import flask

app = Flask(__name__)
clf = pickle.load(open('clf.pkl','rb'))
loaded_vec =pickle.load(open("count_vect.pkl","rb"))

@app.route('/')
def symptom():
    return flask.render_template('index.html')


@app.route('/result',methods =['POST','GET'])
def result():
    if request.method == 'POST':
        result=request.form['Data']
        result_pred=clf.predict(loaded_vec.transform([result]))
        return render_template("predict.html",result = result_pred)
    

if __name__ == "__main__":
    app.run(debug=True)