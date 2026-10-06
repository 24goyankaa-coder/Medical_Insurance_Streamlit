import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Medical Insurance Cost Prediction")

st.write("Enter the details below to predict medical insurance cost.")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["female", "male"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

smoker = st.selectbox(
    "Smoker",
    ["no", "yes"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    input_data = pd.get_dummies(
        input_data,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int
    )

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Medical Insurance Cost: ₹{prediction:,.2f}"
    )