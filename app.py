import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel=pickle.load(open('regmodel.pkl','rb')) ##Loading model
scalr=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predic_api',methods=['POST'])
def predic_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalr.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predic():
    data=[float(x) for x in request.form.values()]
    final_input=scalr.transform(np.array(data).reshape(1,-1))
    print(final_input)
    op = regmodel.predict(final_input)[0]

   # return redirect(url_for('home', pred_text=pred_text))
    return render_template('home.html',prediction_text='Predicted House price: {}'.format(op), show_modal=True)

if __name__=="__main__":
    app.run(debug=True)