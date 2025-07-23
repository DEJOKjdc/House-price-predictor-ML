from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained pipeline (Ridge Regression model)
pipe = pickle.load(open('RidgeMode1.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        sqft = float(request.form['sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])

        # Create DataFrame for prediction
        input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                                columns=['location', 'total_sqft', 'bath', 'bhk'])

        # Predict using the pipeline
        prediction = pipe.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Estimated Price: ₹ {prediction} Lakhs")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
