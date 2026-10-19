
import pandas as pd
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

# 2. Select input features and target
X = df[["study_hours", "attendance"]]
y = df["exam_score"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Train the machine learning model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# 6. Predict a student's exam score
new_student = pd.DataFrame({
    "study_hours": [5],
    "attendance": [80]
})

predicted_score = model.predict(new_student)

print("\nStudent study hours:", 5)
print("Student attendance:", 80)
print("Predicted exam score:", round(predicted_score[0], 2))
