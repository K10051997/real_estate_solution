import streamlit as st
import pandas as pd
import joblib

model = joblib.load("real_estate_solution/models/model.pkl")

st.title("🏠 Real Estate Price Prediction")

st.write("Enter the following inputs:")

# Adjust these inputs based on your dataset's feature columns
col1 = st.number_input("Feature 1", min_value=0.0)
col2 = st.number_input("Feature 2", min_value=0.0)
# Add more inputs as per your feature columns

if st.button("Predict"):
    data = pd.DataFrame([[col1, col2]], columns=["Feature1", "Feature2"])  # Update with actual names
    prediction = model.predict(data)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
