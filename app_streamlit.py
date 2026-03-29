import streamlit as st
import joblib
from src.prediction import predict
from src.data_preprocessing import preprocess_input

# Page config
st.set_page_config(
    page_title="Enes Rain Prediction App",
    page_icon="🌧️",
    layout="centered"
)

# Title
st.title("🌧️ Rain Prediction System")
st.markdown("Predict whether it will rain based on weather conditions.")

# Load encoder
encoder = joblib.load("model/location_encoder.pkl")
locations = list(encoder.classes_)

# Input section
st.header("📥 Enter Weather Details")

location = st.selectbox("🌍 Location", locations)
temperature = st.number_input("🌡️ Temperature", value=25.0)
humidity = st.slider("💧 Humidity (%)", 0, 100, 60)
wind_speed = st.number_input("🌬️ Wind Speed", value=10.0)
precipitation = st.number_input("🌧️ Precipitation", value=0.0)
cloud_cover = st.slider("☁️ Cloud Cover (%)", 0, 100, 40)
pressure = st.number_input("📊 Pressure", value=1012.0)

# Predict button
if st.button("🔮 Predict Rain", use_container_width=True):

    input_data = [
        location,
        temperature,
        humidity,
        wind_speed,
        precipitation,
        cloud_cover,
        pressure
    ]

    processed_data = preprocess_input(input_data)
    prediction = predict(processed_data)

    st.markdown("---")

    # Output
    if prediction == 1:
        st.error("🌧️ It WILL rain!")
    else:
        st.success("☀️ No rain expected.")

# Footer
st.markdown("---")
st.caption("Built By Enes  with ❤️ using Machine Learning and Streamlit")