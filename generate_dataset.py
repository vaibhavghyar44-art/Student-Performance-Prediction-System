from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "student_data.csv"

def generate_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    number_of_students = 500

    attendance = rng.integers(40, 101, number_of_students)
    study_hours = rng.uniform(0, 8, number_of_students).round(1)
    assignment_score = rng.integers(20, 101, number_of_students)
    previous_score = rng.integers(20, 101, number_of_students)

    noise = rng.normal(0, 5, number_of_students)

    final_score = (
        0.20 * attendance 
        + 2.5 * study_hours
        + 0.25 * assignment_score
        + 0.35 * previous_score
        + noise
    )

    final_score = np.clip(final_score, 0, 100).round(1)

    data = pd.DataFrame({
        "attendance": attendance,
        "study_hours": study_hours,
        "assignment_score": assignment_score,
        "previous_score": previous_score,
        "final_score": final_score
    })

    data.to_csv(DATA_FILE, index=False)

    print("Dataset generated successfully.")
    print(f"Dataset path: {DATA_FILE}")
    print(f"Number of records: {len(data)}")
    print(data.head())


if __name__ == "__main__":
    generate_dataset()