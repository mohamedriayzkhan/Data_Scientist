import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="KNN Prediction", layout="centered")

# Get absolute path of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler safely
model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))


st.set_page_config(page_title="KNN Prediction", layout="centered")

st.title("🧠 KNN Purchase Prediction App")
st.write("Predict whether a customer will purchase or not")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100, value=30)
salary = st.number_input("Estimated Salary", min_value=0, value=50000)

if st.button("Predict"):
    input_data = np.array([[age, salary]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Customer is likely to PURCHASE")
    else:
        st.warning("❌ Customer is NOT likely to purchase")
