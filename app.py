import streamlit as st
from src.prediction import predict
from src.data_preprocessing import preprocess_input

st.title("🌧️ Rain Prediction App")

# Input fields
location = st.number_input("Location (encoded)", 0, 100, 0)
temperature = st.number_input("Temperature", value=25.0)
humidity = st.number_input("Humidity", value=60.0)
wind_speed = st.number_input("Wind Speed", value=10.0)
precipitation = st.number_input("Precipitation", value=0.0)
cloud_cover = st.number_input("Cloud Cover", value=40.0)
pressure = st.number_input("Pressure", value=1012.0)
year = st.number_input("Year", value=2025)
month = st.number_input("Month", 1, 12, 1)
day = st.number_input("Day", 1, 31, 1)

# Predict button
if st.button("Predict Rain"):
    # Collect input
    input_data = [
        location, temperature, humidity, wind_speed,
        precipitation, cloud_cover, pressure,
        year, month, day
    ]

    # Preprocess (currently minimal)
    processed_data = preprocess_input(input_data)

    # Predict
    prediction = predict(processed_data)

    # Output
    if prediction == 1:
        st.success("🌧️ It will rain!")
    else:
        st.info("☀️ No rain predicted")