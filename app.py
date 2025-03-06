import streamlit as st
import pandas as pd
import pickle
import os

# Debugging: Check current directory and files
st.write("Current Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir())

# Load the trained model
model_path = "linear_model.pkl"  # Use relative path
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
else:
    st.error("ERROR: linear_model.pkl NOT FOUND!")

# Streamlit UI
st.title("Medical Treatment Cost Prediction")

# User Inputs
age = st.number_input("Enter Age:", min_value=1, max_value=100, step=1)
treatment_type = st.selectbox("Select Treatment Type:", ["General Checkup", "Surgery", "Medication", "Physical Therapy"])
health_condition = st.selectbox("Select Health Condition:", ["Healthy", "Diabetes", "Hypertension", "Asthma", "Heart Disease"])

# Create a DataFrame with the *same column names* used when training the model
X_input = pd.DataFrame([[age, health_condition, treatment_type]], 
                       columns=['Age', 'Health Condition', 'Treatment Type'])

# Predict cost when button is clicked
if st.button("Predict"):
    try:
        prediction = model.predict(X_input)  # No encoding needed, model will do it
        st.success(f"Predicted Treatment Cost: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")