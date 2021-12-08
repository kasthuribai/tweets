# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:00:57 2021

@author: USER
"""

from flask import Flask, render_template, request
#from joblib import load
import pickle
import os

app = Flask(__name__)
clf = pickle.load(open('clf.pkl','rb'))
loaded_vec =pickle.load(open("count_vect.pkl","rb"))


@app.route('/')
def symptom():
    return render_template('index.html')


@app.route('/result',methods =['POST','GET'])
def result():
    if request.method == 'POST':
        result=request.form['Data']
        result_pred=clf.predict(loaded_vec.transform([result]))
        return render_template("predict.html",result = result_pred)
    

if __name__ == '__main__':     
     port = int(os.environ.get('PORT', 5000))
     #app.debug = True
     #app.run(host='192.168.5.1', port=port)
     app.run(host='0.0.0.0', port=port,use_reloader=False, debug=True)
     #app.run