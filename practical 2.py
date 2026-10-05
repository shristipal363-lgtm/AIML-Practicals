import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Creating the dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 6],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 92, 95, 82],
    "Assignment_Score": [50, 55, 60, 68, 72, 78, 85, 88, 94, 75],
    "Final_Marks": [52, 58, 64, 70, 76, 82, 88, 91, 96, 79]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Display dataset
print("Student Performance Dataset:")
print(df)

# Calculating correlation
print("\nCorrelation Matrix:")
print(df.corr())

# Plot 1: Study Hours vs Final Marks
plt.figure(figsize=(6, 4))
plt.scatter(df["Study_Hours"], df["Final_Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Final Marks")
plt.grid()
plt.show()

# Plot 2: Attendance vs Final Marks
plt.figure(figsize=(6, 4))
plt.scatter(df["Attendance"], df["Final_Marks"])
plt.xlabel("Attendance (%)")
plt.ylabel("Final Marks")
plt.title("Attendance vs Final Marks")
plt.grid()
plt.show()

# Plot 3: Assignment Score vs Final Marks
plt.figure(figsize=(6, 4))
plt.scatter(df["Assignment_Score"], df["Final_Marks"])
plt.xlabel("Assignment Score")
plt.ylabel("Final Marks")
plt.title("Assignment Score vs Final Marks")
plt.grid()
plt.show()

# Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()