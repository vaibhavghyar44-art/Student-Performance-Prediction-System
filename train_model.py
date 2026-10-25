from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "student_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "student_model.pkl"

FEATURES = [
    "attendance",
    "study_hours",
    "assignment_score",
    "previous_score"
]

TARGET = "final_score"

def train_model():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Run generate_dataset.py first."
        )
    
    data = pd.read_csv(DATA_FILE)

    # EDA 
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset column: {missing_columns}"
        )
    data = data[required_columns].dropna()

    if len(data) < 10:
        raise ValueError("At least 10 valid records are required.")

    x = data[FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    # Create the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    # Model evaluation 
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Create folder to save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": model,
        "feature": FEATURES,
        "MAE": float(mae),
        "r2": float(r2)
    }
    with MODEL_FILE.open("wb") as file:
        pickle.dump(model_data, file)

    print("\nModel training completed successfully!")
    print("Training records:", len(x_train))
    print("Testing records:", len(x_test))
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.3f}")
    print("Model Saved at:", MODEL_FILE)

if __name__ == "__main__":
    train_model()