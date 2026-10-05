import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Create dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 4, 8],
    "Attendance": [55, 60, 65, 70, 75, 80, 85, 90, 95, 78, 62, 88],
    "Assignment_Score": [45, 50, 55, 60, 65, 70, 78, 85, 94, 68, 52, 80],
    "Final_Marks": [52, 58, 64, 70, 76, 82, 88, 91, 96, 79, 63, 89]
}

df = pd.DataFrame(data)

print("Student Performance Dataset:")
print(df)


# Step 2: Separate input and output
X = df[["Study_Hours", "Attendance", "Assignment_Score"]]
y = df["Final_Marks"]


# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# Step 4: Standardize the data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 5: Create Ridge Regression model
ridge_model = Ridge(alpha=1.0)


# Step 6: Create Lasso Regression model
lasso_model = Lasso(alpha=0.1)


# Step 7: Train Ridge model
ridge_model.fit(X_train_scaled, y_train)


# Step 8: Train Lasso model
lasso_model.fit(X_train_scaled, y_train)


# Step 9: Make predictions
ridge_pred = ridge_model.predict(X_test_scaled)
lasso_pred = lasso_model.predict(X_test_scaled)


# Step 10: Evaluate Ridge Regression
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print("\n----- Ridge Regression -----")

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(ridge_pred)

print("\nMean Squared Error:")
print(ridge_mse)

print("\nR2 Score:")
print(ridge_r2)


# Step 11: Evaluate Lasso Regression
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print("\n----- Lasso Regression -----")

print("\nPredicted Values:")
print(lasso_pred)

print("\nMean Squared Error:")
print(lasso_mse)

print("\nR2 Score:")
print(lasso_r2)


# Step 12: Display coefficients
print("\n----- Model Coefficients -----")

print("\nRidge Coefficients:")
print(ridge_model.coef_)

print("\nLasso Coefficients:")
print(lasso_model.coef_)


# Step 13: Predict a new student
new_student = pd.DataFrame(
    [[7, 82, 72]],
    columns=["Study_Hours", "Attendance", "Assignment_Score"]
)

new_student_scaled = scaler.transform(new_student)

ridge_prediction = ridge_model.predict(new_student_scaled)
lasso_prediction = lasso_model.predict(new_student_scaled)

print("\n----- New Student Prediction -----")

print("\nRidge Prediction:")
print(ridge_prediction[0])

print("\nLasso Prediction:")
print(lasso_prediction[0])