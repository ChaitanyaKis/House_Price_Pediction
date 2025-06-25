from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    try:
        data = [
            float(request.form['area']),
            int(request.form['bedrooms']),
            int(request.form['bathrooms']),
            int(request.form['stories']),
            int(request.form['parking']),
            int(request.form['mainroad']),
            int(request.form['guestroom']),
            int(request.form['basement']),
            int(request.form['hotwaterheating']),
            int(request.form['airconditioning']),
            int(request.form['prefarea']),
            int(request.form['furnishingstatus'])
        ]

        
        final_input = scaler.transform([data])
        prediction = model.predict(final_input)[0]

        return render_template('index.html', prediction_text=f'Predicted Price: ₹{round(prediction, 2)} Lakhs')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
