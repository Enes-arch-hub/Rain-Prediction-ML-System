import joblib

def load_model(path="model/rainfall_model.pkl"):
    return joblib.load(path)