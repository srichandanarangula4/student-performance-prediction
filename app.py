
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓"
)

# App title
st.title("🎓 Student Performance Predictor")
st.write("Predict a student's exam score using study hours and attendance.")

# Sample training data
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 9],
    "attendance": [60, 65, 70, 75, 80, 85, 90, 95, 55, 72, 88, 98],
    "exam_score": [35, 40, 48, 55, 65, 70, 78, 88, 38, 53, 73, 92]
}

df = pd.DataFrame(data)

# Prepare the machine-learning model
X = df[["study_hours", "attendance"]]
y = df["exam_score"]

model = LinearRegression()
model.fit(X, y)

# User inputs
st.subheader("Enter Student Details")
study_hours = st.slider(
    "Daily study hours",
    min_value=0,
    max_value=12,
    value=5
)

attendance = st.slider(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=80
)

# Predict score when the button is clicked

if st.button("Predict Exam Score"):
    student = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance]
    })

    predicted_score = model.predict(student)[0]
    predicted_score = max(0, min(100, predicted_score))

    st.success(f"Predicted Exam Score: {predicted_score:.2f} / 100")
    st.write(f"**Daily study hours:** {study_hours} hours")
    st.write(f"**Attendance:** {attendance}%")

    
    # Dynamic prediction summary
    st.subheader("Prediction Summary")

    st.metric(
        label="Predicted Exam Score",
        value=f"{predicted_score:.2f}/100"
    )

    if predicted_score >= 80:
        st.write("The model predicts a score in the 80–100 range.")
    elif predicted_score >= 60:
        st.write("The model predicts a score in the 60–79.99 range.")
    elif predicted_score >= 40:
        st.write("The model predicts a score in the 40–59.99 range.")
    else:
        st.write("The model predicts a score below 40.")

    st.caption(
        "This summary describes the model's output, "
        "not a definitive assessment of a student's ability."
    )


    # Display a progress bar for the predicted score
    st.write("### Predicted Score Progress")
    st.progress(int(round(predicted_score)))

    st.caption(
        f"The predicted score is {predicted_score:.2f}% "
        "on a scale of 0 to 100."
    )

    st.info("This is an educational demo using a small synthetic dataset.")
    
    # Show how study hours relate to exam scores
    st.write("### Study Hours vs Exam Scores")

    fig, ax = plt.subplots()

    ax.scatter(
        df["study_hours"],
        df["exam_score"],
        label="Sample student scores"
    )

    ax.scatter(
        [study_hours],
        [predicted_score],
        marker="*",
        s=250,
        label="Your predicted score"
    )

    ax.set_xlabel("Daily Study Hours")
    ax.set_ylabel("Exam Score")
    ax.set_title("Study Hours and Exam Performance")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)
