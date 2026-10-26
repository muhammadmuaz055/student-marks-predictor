
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("🎓 Student Marks Predictor")
st.write("Predict student marks using Machine Learning.")

# Sample training data
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
marks = np.array([20, 30, 40, 50, 60, 70, 80, 90])

# Train the ML model
model = LinearRegression()
model.fit(hours, marks)

# User input
study_hours = st.slider(
    "How many hours do you study?",
    min_value=0,
    max_value=10,
    value=5
)

if st.button("Predict Marks"):
    prediction = model.predict(
        np.array([[study_hours]])
    )[0]

    prediction = np.clip(prediction, 0, 100)

    st.success(f"Predicted Marks: {prediction:.2f}/100")
