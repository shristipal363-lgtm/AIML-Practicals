import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Create dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 3, 8, 4, 7],
    "Attendance": [55, 60, 65, 70, 75, 80, 85, 90, 58, 88, 62, 82],
    "Assignment_Score": [45, 50, 55, 60, 65, 70, 78, 85, 48, 80, 52, 72],
    "Result": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Step 2: Separate input and output
X = df[["Study_Hours", "Attendance", "Assignment_Score"]]
y = df["Result"]

# Step 3: Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 4: Standardize the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Step 6: Train the model
knn.fit(X_train, y_train)

# Step 7: Predict test data
y_pred = knn.predict(X_test)

# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

print("\nAccuracy:")
print(accuracy)

# Step 9: Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 10: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 11: Predict a new student
new_student = pd.DataFrame(
    [[6, 78, 68]],
    columns=["Study_Hours", "Attendance", "Assignment_Score"]
)

new_student_scaled = scaler.transform(new_student)

prediction = knn.predict(new_student_scaled)

if prediction[0] == 1:
    print("\nNew Student Result: PASS")
else:
    print("\nNew Student Result: FAIL")