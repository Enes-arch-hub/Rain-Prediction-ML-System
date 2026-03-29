import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(data_path, model_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Drop Date column if it exists
    if "Date" in df.columns:
        df = df.drop("Date", axis=1)

    # Encode Location
    le = LabelEncoder()
    df["Location"] = le.fit_transform(df["Location"])

    # Save encoder
    joblib.dump(le, "model/location_encoder.pkl")

    # Features and target
    X = df.drop("Rain Tomorrow", axis=1)
    y = df["Rain Tomorrow"]

    print("Columns used:", X.columns)
    print("Shape:", X.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model with overfitting control
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.4f}")

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # 🔍 Feature Importance (IMPORTANT FOR DEBUGGING)
    print("\n🔍 Feature Importance:\n")
    for name, importance in zip(X.columns, model.feature_importances_):
        print(f"{name}: {importance:.4f}")

    # Save model
    joblib.dump(model, model_path)

    print("✅ Model and encoder saved!")