from flask import Flask, render_template, request
import numpy as np
from joblib import dump,load

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == "POST":
		email = request.form["email"]
	if email == "":
		res = {'null':'null'}
	else:
		vect = load('vectorizer.pkl')
		scaler = load('scaler.pkl')
		lgr = load('model.pkl')
		text_v = vect.transform([email])
		text_std = scaler.transform(text_v)
		prediction = lgr.predict(text_std)
		prob = lgr.predict_proba(text_std)
		if prediction == 0:
			res = {"ham":int(np.round(prob[0][0]*100))}
		elif prediction == 1:
			res = {"spam":int(np.round(prob[0][1]*100))}
	return render_template('index.html',pred=res.items())
		 

if __name__ == "__main__":
	app.run(debug=True) 