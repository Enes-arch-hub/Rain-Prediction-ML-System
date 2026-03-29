import numpy as np
import joblib

model = joblib.load("model/rain_model.pkl")
encoder = joblib.load("model/location_encoder.pkl")

def predict(input_data):
    # Encode location
    input_data[0] = encoder.transform([input_data[0]])[0]

    # Convert to numpy
    input_array = np.array(input_data).reshape(1, -1)

    return model.predict(input_array)[0]