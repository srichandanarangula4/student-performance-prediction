import pandas as pd

# Sample dataset for learning
data = {
    "study_hours": [1, 2, 3, 4, 5, 6],
    "attendance": [60, 65, 75, 80, 90, 95],
    "exam_score": [40, 45, 60, 65, 80, 90],
}

df = pd.DataFrame(data)

print("Student Performance Dataset")
print(df)

print("\nAverage exam score:")
print(df["exam_score"].mean())

print("\nDataset summary:")
print(df.describe())
