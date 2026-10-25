from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

from recommendations import get_recommendations


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "models" / "student_model.pkl"
DATA_FILE = BASE_DIR / "data" / "student_data.csv"


# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    with MODEL_FILE.open("rb") as file:
        return pickle.load(file)


# ==========================================
# 4. APPLICATION HEADER
# ==========================================

st.title("🎓 Student Performance Prediction System")

st.write(
    "Predict academic performance and receive personalized "
    "learning recommendations based on student details."
)

st.info(
    "This demonstration uses synthetic sample data. "
    "Predictions are estimates, not guaranteed exam results."
)


# ==========================================
# 5. CHECK MODEL FILE
# ==========================================

if not MODEL_FILE.exists():
    st.error(
        "Trained model not found. Please run "
        "generate_dataset.py and train_model.py first."
    )
    st.stop()


# ==========================================
# 6. LOAD MODEL AND FEATURES
# ==========================================

try:
    model_data = load_model()

    model = model_data["model"]
    # Handled both possibilities to prevent KeyError depending on training script version
    features = model_data.get("features", model_data.get("feature"))

except (OSError, pickle.UnpicklingError, EOFError, KeyError) as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ==========================================
# 7. STUDENT INPUT FORM
# ==========================================

st.subheader("📝 Enter Student Details")

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

    with col2:

        assignment_score = st.number_input(
            "Assignment Score (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

        previous_score = st.number_input(
            "Previous Exam Score (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    submitted = st.form_submit_button(
        "Predict Performance",
        type="primary",
        use_container_width=True
    )


# ==========================================
# 8. PREDICTION AND RECOMMENDATIONS
# ==========================================

if submitted:

    try:
        # Prepare input data
        input_data = pd.DataFrame(
            [[
                attendance,
                study_hours,
                assignment_score,
                previous_score
            ]],
            columns=features
        )

        # Match the feature order used during model training
        if features is not None:
            student_data = input_data[features]
        else:
            student_data = input_data

        # Predict final score
        predicted_score = float(
            model.predict(student_data)[0]
        )

        # Keep the score within 0 to 100
        predicted_score = max(
            0.0,
            min(100.0, predicted_score)
        )

        # Determine performance category
        if predicted_score < 40:
            category = "Needs Improvement"

        elif predicted_score < 70:
            category = "Average"

        else:
            category = "Good"

        # Display results
        st.divider()
        st.subheader("📊 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Final Score",
                f"{predicted_score:.1f} / 100"
            )

        with col2:
            st.metric(
                "Performance Category",
                category
            )

        # Display score progress
        st.write("Predicted Score Progress")

        st.progress(
            int(round(predicted_score))
        )

        # Display personalized recommendations
        st.subheader("📚 Personalized Learning Recommendations")

        recommendations = get_recommendations(
            attendance=attendance,
            study_hours=study_hours,
            assignment_score=assignment_score,
            previous_score=previous_score,
            predicted_score=predicted_score
        )

        if recommendations:
            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")
        else:
            st.write(
                "Keep up the good work! Continue following "
                "your current study routine."
            )

    except (ValueError, KeyError, TypeError, AttributeError) as error:
        st.error(
            f"Unable to generate prediction. Please check "
            f"your model configuration. Details: {error}"
        )


# ==========================================
# 9. DATASET ANALYSIS
# ==========================================

st.divider()
st.subheader("📈 Dataset Analysis")

if DATA_FILE.exists():

    try:
        data = pd.read_csv(DATA_FILE)

        # Check required dataset columns
        required_columns = [
            "attendance",
            "study_hours",
            "final_score"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            st.error(
                "The dataset is missing these required columns: "
                + ", ".join(missing_columns)
            )

        else:

            # ------------------------------------------
            # 9.1 DISPLAY FIRST 10 RECORDS
            # ------------------------------------------

            col1, col2 = st.columns(2)

            with col1:
                st.write("First 10 Student Records")

                st.dataframe(
                    data.head(10),
                    use_container_width=True
                )

            # ------------------------------------------
            # 9.2 STUDY HOURS VS FINAL SCORE
            # ------------------------------------------

            with col2:
                st.write("Study Hours vs Final Score")

                st.scatter_chart(
                    data,
                    x="study_hours",
                    y="final_score",
                    use_container_width=True
                )

            # ------------------------------------------
            # 9.3 AVERAGE SCORE BY ATTENDANCE RANGE
            # ------------------------------------------

            st.subheader(
                "Average Score by Attendance Range"
            )

            # Include attendance values from 0 to 100
            data["attendance_range"] = pd.cut(
                data["attendance"],
                bins=[
                    -0.001,
                    50,
                    60,
                    70,
                    80,
                    90,
                    100
                ],
                labels=[
                    "0-50%",
                    "50-60%",
                    "60-70%",
                    "70-80%",
                    "80-90%",
                    "90-100%"
                ],
                include_lowest=True
            )

            # Calculate average final score for each range
            average_scores = (
                data.groupby(
                    "attendance_range",
                    observed=False
                )["final_score"]
                .mean()
                .dropna()
            )

            if not average_scores.empty:

                # Convert categorical index to string labels
                average_scores_df = (
                    average_scores
                    .rename("Average Score")
                    .rename_axis("Attendance Range")
                    .reset_index()
                )

                average_scores_df["Attendance Range"] = (
                    average_scores_df["Attendance Range"].astype(str)
                )

                # Display the bar chart
                st.bar_chart(
                    average_scores_df,
                    x="Attendance Range",
                    y="Average Score",
                    use_container_width=True
                )

            else:
                st.warning(
                    "No valid attendance data is available "
                    "to generate the bar chart."
                )

    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(
            f"Unable to analyze the dataset: {error}"
        )

else:
    st.warning(
        "Dataset not found. Please run generate_dataset.py first."
    )


# ==========================================
# 10. SIDEBAR: MODEL INFORMATION
# ==========================================

st.sidebar.title("🤖 Model Information")

st.sidebar.write(
    "Algorithm: Random Forest Regressor"
)

if "mae" in model_data:

    st.sidebar.metric(
        "Test MAE",
        f"{model_data['mae']:.2f}"
    )

if "r2" in model_data:

    st.sidebar.metric(
        "Test R² Score",
        f"{model_data['r2']:.3f}"
    )

st.sidebar.caption(
    "Evaluation metrics are based on synthetic sample data."
)

st.sidebar.divider()

st.sidebar.write(
    "Student Performance Prediction System"
)

st.sidebar.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)