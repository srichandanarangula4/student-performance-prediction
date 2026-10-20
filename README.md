# Student Performance Prediction Using Machine Learning

## Project Overview
This project uses a Linear Regression model to predict student exam scores based on study hours and attendance.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Linear Regression
- Model Evaluation

## Dataset
A small synthetic dataset containing:
- Study hours
- Attendance percentage
- Exam scores

**Note:** The dataset is synthetic and is intended for educational purposes only.

## Project Workflow
1. Created a sample dataset using Pandas.
2. Selected study hours and attendance as input features.
3. Split the data into training and testing sets.
4. Trained a Linear Regression model.
5. Evaluated model performance using MAE and R².
6. Generated a prediction for a new student.

## Results

- **Mean Absolute Error (MAE):** 1.78
- **R² Score:** 0.9862
- **Example prediction:** 63.29 for a student studying 5 hours with 80% attendance.

These results are based on a very small synthetic dataset and should not be interpreted as real-world model performance.

## How to Run

Install the required libraries:

```bash
py -m pip install pandas scikit-learn
```

Run the project:

```bash
py student_performance.py
```

## Future Improvements
- Visualize study hours and exam scores.
- Evaluate the model on a larger dataset.
- Compare multiple machine learning algorithms.
- Build an interactive prediction interface.

## Author
Srichandana Rangula
