import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('mlr_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit application
st.title('Set Temperature Prediction')

# Input fields for user input
no_of_pers = st.number_input('Number of Persons', min_value=0.0, format="%.1f")
temperature = st.number_input('Temperature', min_value=0.0, max_value=50.0, format="%.1f")
humidity = st.number_input('Humidity', min_value=0.0, max_value=100.0, format="%.1f")

# Predict button
if st.button('Predict'):
    # Prepare input data for the model
    input_data = np.array([[no_of_pers, temperature, humidity]])
    
    # Make prediction
    prediction = round(model.predict(input_data)[0], 1)
    
    # Display prediction
    st.success(f'The predicted set temperature is: {prediction}°C')
