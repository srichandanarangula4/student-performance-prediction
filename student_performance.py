
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create sample student data
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 9],
    "attendance": [60, 65, 70, 75, 80, 85, 90, 95, 55, 72, 88, 98],
    "exam_score": [35, 40, 48, 55, 65, 70, 78, 88, 38, 53, 73, 92]
}

df = pd.DataFrame(data)

# 2. Select features and target
X = df[["study_hours", "attendance"]]
y = df["exam_score"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# 6. Predict for a new student
new_student = pd.DataFrame({
    "study_hours": [5],
    "attendance": [80]
})

predicted_score = model.predict(new_student)

print("\nStudy hours: 5")
print("Attendance: 80%")
print("Predicted exam score:", round(predicted_score[0], 2))

# 7. Create charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart 1: Study hours versus exam scores
axes[0].scatter(
    df["study_hours"], df["exam_score"], label="Student scores"
)
axes[0].set_title("Study Hours vs Exam Scores")
axes[0].set_xlabel("Study Hours")
axes[0].set_ylabel("Exam Score")
axes[0].legend()
axes[0].grid(True)

# Chart 2: Actual versus predicted test scores
axes[1].scatter(
    y_test, predictions, label="Predictions"
)

# Reference line for perfect predictions
axes[1].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    label="Perfect prediction"
)

axes[1].set_title("Actual vs Predicted Scores")
axes[1].set_xlabel("Actual Score")
axes[1].set_ylabel("Predicted Score")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

# 8. Save the charts in your project folder
plt.savefig("student_performance_charts.png", dpi=150)

print("\nChart saved as student_performance_charts.png")

# 9. Display the charts
plt.show()
