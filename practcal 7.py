import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Create dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 6],
    "Final_Marks": [52, 58, 64, 70, 76, 82, 88, 91, 96, 79]
}

df = pd.DataFrame(data)

print("Student Performance Dataset:")
print(df)


# Step 2: Separate input and output
X = df[["Study_Hours"]]
y = df["Final_Marks"]


# Step 3: Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Standardize the input data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 5: Create SVR model
model = SVR(
    kernel="rbf",
    C=100,
    epsilon=0.1
)


# Step 6: Train the model
model.fit(X_train_scaled, y_train)


# Step 7: Predict test data
y_pred = model.predict(X_test_scaled)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)


# Step 8: Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error:")
print(mse)


# Step 9: Calculate R2 Score
r2 = r2_score(y_test, y_pred)

print("\nR2 Score:")
print(r2)


# Step 10: Predict marks for a new student
new_student = pd.DataFrame(
    [[7]],
    columns=["Study_Hours"]
)

new_student_scaled = scaler.transform(new_student)

predicted_marks = model.predict(new_student_scaled)

print("\nPredicted Marks for a Student Studying 7 Hours:")
print(predicted_marks[0])


# Step 11: Plot actual data
plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

# Create values for smooth prediction line
X_plot = pd.DataFrame(
    [[i] for i in range(2, 11)],
    columns=["Study_Hours"]
)

X_plot_scaled = scaler.transform(X_plot)

y_plot = model.predict(X_plot_scaled)

plt.plot(
    X_plot,
    y_plot,
    label="SVR Prediction"
)

plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Support Vector Regression")
plt.legend()
plt.grid()

plt.show()