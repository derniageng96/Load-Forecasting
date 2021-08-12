import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

#model = pickle.load(open('model.pkl', 'rb'))
model = load_model('model')

app = Flask(__name__)
@app.route('/')
def home():
      return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]

    final_features = np.array(int_features)
    final_features = [final_features.reshape((1, 9, 5))]
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='Predicted kWh: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)