from flask import Flask, request, render_template
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load the trained model
with open('mlr_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get form data
        no_of_pers = float(request.form['no_of_persons'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])

        # Prepare input data for the model
        input_data = np.array([[no_of_pers, temperature, humidity]])
        
        # Make prediction
        prediction = round(model.predict(input_data)[0], 1)

    return render_template('input_form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
