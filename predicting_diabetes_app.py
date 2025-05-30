import streamlit as st
import pandas as pd
import numpy as np
import pickle
# Add any other libraries your app specifically uses for prediction (e.g., sklearn.preprocessing if you use scalers)

# --- Model Loading (Crucial Part) ---
# Ensure 'diabetes_model.pkl' is correctly saved from your trained model and downloaded
try:
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # You can add a Streamlit message for debugging if needed: st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'diabetes_model.pkl' not found. Please ensure it's in the same directory as your app.")
    st.stop() # Stop the app if the model isn't found
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop()
# --- End Model Loading ---


# --- Streamlit Interface Code ---
st.title('🩺 Diabetes Prediction App')
st.write('This app predicts the likelihood of diabetes based on patient health metrics.')

st.sidebar.header('Patient Input Features')

def user_input_features():
    # Make sure these match the features your model was trained on, in the correct order
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.471)
    age = st.sidebar.slider('Age', 21, 81, 33)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Features')
st.write(df)

st.subheader('Prediction')

if st.button('Predict Diabetes'):
    try:
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        st.markdown('---')
        if prediction[0] == 1:
            st.error('**Prediction: The patient is likely to have Diabetes.**')
        else:
            st.success('**Prediction: The patient is likely NOT to have Diabetes.**')

        st.write(f"Confidence (No Diabetes): {prediction_proba[0][0]*100:.2f}%")
        st.write(f"Confidence (Diabetes): {prediction_proba[0][1]*100:.2f}%")
        st.markdown('---')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check the input values or contact support if the issue persists.")

st.sidebar.markdown('---')
st.sidebar.info('This app uses a machine learning model to predict diabetes. It is for informational purposes only and should not be used for medical diagnosis.')
