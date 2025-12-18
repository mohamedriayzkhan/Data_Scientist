import streamlit as st
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

st.title("🧠 KNN Purchase Prediction")

age = st.number_input("Age", 1, 100, 30)
salary = st.number_input("Estimated Salary", 0, 200000, 50000)

if st.button("Predict"):
    data = np.array([[age, salary]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)

    if pred[0] == 1:
        st.success("✅ Customer will PURCHASE")
    else:
        st.warning("❌ Customer will NOT purchase")
