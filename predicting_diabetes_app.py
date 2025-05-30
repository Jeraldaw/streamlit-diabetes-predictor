
import streamlit as st
import pandas as pd
import numpy as np
# Assuming you have a pre-trained model saved as 'diabetes_model.pkl'
import pickle

# Load your pre-trained model (if applicable)
try:
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'diabetes_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop() # Stop the app if the model isn't found

st.title('Diabetes Prediction App')
st.write('Enter patient details to predict diabetes.')

# Input fields for features (example features, adjust based on your model)
pregnancies = st.slider('Pregnancies', 0, 17, 3)
glucose = st.slider('Glucose', 0, 200, 120)
blood_pressure = st.slider('Blood Pressure', 0, 122, 70)
skin_thickness = st.slider('Skin Thickness', 0, 99, 20)
insulin = st.slider('Insulin', 0, 846, 79)
bmi = st.slider('BMI', 0.0, 67.1, 32.0)
diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.471)
age = st.slider('Age', 21, 81, 33)

# Create a DataFrame for prediction
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                          columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# Prediction button
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader('Prediction Results:')
        if prediction[0] == 1:
            st.error('The patient is predicted to have Diabetes.')
        else:
            st.success('The patient is predicted not to have Diabetes.')

        st.write(f'Probability of Diabetes: {prediction_proba[0][1]:.2f}')
        st.write(f'Probability of No Diabetes: {prediction_proba[0][0]:.2f}')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.sidebar.header('About')
st.sidebar.info('This app predicts diabetes based on input features using a machine learning model.')
