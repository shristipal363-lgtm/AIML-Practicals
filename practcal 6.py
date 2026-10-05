import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Create dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 6],
    "Final_Marks": [52, 58, 64, 70, 76, 82, 88, 91, 96, 79]
}

df = pd.DataFrame(data)

print("Student Performance Dataset:")
print(df)


# Step 2: Separate independent and dependent variables
X = df[["Study_Hours"]]
y = df["Final_Marks"]


# Step 3: Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Create Linear Regression model
model = LinearRegression()


# Step 5: Train the model
model.fit(X_train, y_train)


# Step 6: Predict test data
y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)


# Step 7: Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error:")
print(mse)


# Step 8: Calculate R2 Score
r2 = r2_score(y_test, y_pred)

print("\nR2 Score:")
print(r2)


# Step 9: Display coefficient and intercept
print("\nCoefficient:")
print(model.coef_[0])

print("\nIntercept:")
print(model.intercept_)


# Step 10: Predict marks for a new student
new_student = pd.DataFrame(
    [[7]],
    columns=["Study_Hours"]
)

predicted_marks = model.predict(new_student)

print("\nPredicted Marks for a Student Studying 7 Hours:")
print(predicted_marks[0])


# Step 11: Plot the regression line
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Study_Hours"],
    df["Final_Marks"],
    label="Actual Data"
)

plt.plot(
    df["Study_Hours"],
    model.predict(df[["Study_Hours"]]),
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Linear Regression - Study Hours vs Final Marks")
plt.legend()
plt.grid()

plt.show()