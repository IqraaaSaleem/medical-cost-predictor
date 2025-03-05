import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("linear_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Medical Treatment Cost Prediction")

# User Inputs
age = st.number_input("Enter Age:", min_value=1, max_value=100, step=1)
treatment_type = st.selectbox("Select Treatment Type:", ["General Checkup", "Surgery", "Medication", "Physical Therapy"])
health_condition = st.selectbox("Select Health Condition:", ["Healthy", "Diabetes", "Hypertension", "Asthma", "Heart Disease"])

# Create a DataFrame with the **same column names** used when training the model
X_input = pd.DataFrame([[age, health_condition, treatment_type]], 
                       columns=['Age', 'Health Condition', 'Treatment Type'])

# Predict cost when button is clicked
if st.button("Predict"):
    try:
        prediction = model.predict(X_input)  # No encoding needed, model will do it
        st.success(f"Predicted Treatment Cost: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
