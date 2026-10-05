import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Creating the dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 3, 8, 4, 7],
    "Attendance": [55, 60, 65, 70, 75, 80, 85, 90, 58, 88, 62, 82],
    "Assignment_Score": [45, 50, 55, 60, 65, 70, 78, 85, 48, 80, 52, 72],
    "Result": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1]
}

# Creating DataFrame
df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Selecting independent variables
X = df[["Study_Hours", "Attendance", "Assignment_Score"]]

# Selecting dependent variable
y = df["Result"]

# Splitting dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Creating Naive Bayes classifier
model = GaussianNB()

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

print("\nAccuracy:")
print(accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Predicting result for a new student
new_student = [[6, 78, 68]]

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nNew Student Result: PASS")
else:
    print("\nNew Student Result: FAIL")