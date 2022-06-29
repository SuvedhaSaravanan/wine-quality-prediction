

from flask import Flask,render_template,json,jsonify,request
import pickle
import numpy as np
import sklearn




app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        alcohol=float(request.form['alcohol'])
        sulphates=float(request.form['sulphates'])
        density=float(request.form['density'])
        citric_acid=float(request.form['citric_acid'])
        volatile_acidity=float(request.form['volatile_acidity'])
        
        filename='model.pickle'
        loaded_model=pickle.load(open(filename,'rb'))
        data=np.array([[volatile_acidity,citric_acid,density,sulphates,alcohol]])
        my_prediction=loaded_model.predict(data)
       
        return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
