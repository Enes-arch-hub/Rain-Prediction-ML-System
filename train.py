from src.model_training import train_model

if __name__ == "__main__":
    train_model(
        data_path="data/rain_data.csv",
        model_path="model/rain_model.pkl"
    )