import pickle
from flask import Flask,request,app,jsonify,render_template
import numpy as np

app = Flask(__name__) # tạo một đối tượng Flask
model = pickle.load(open('housepred.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    output = round(output, 2)
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))


if __name__ =="__main__":
    app.run(debug=True)