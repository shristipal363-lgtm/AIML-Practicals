import pandas as pd
import numpy as np

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Create dataset
data = {
    "Text": [
        "Government announces new education policy",
        "Scientists discover new method to treat disease",
        "New technology improves online education",
        "Aliens have landed in the city",
        "Drinking this magical liquid gives instant superpowers",
        "Celebrity claims moon is made of cheese",
        "University announces new examination schedule",
        "Researchers publish new climate study",
        "Government launches new student scholarship",
        "Secret medicine can make humans live forever"
    ],

    "Image": [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg",
        "image4.jpg",
        "image5.jpg",
        "image6.jpg",
        "image7.jpg",
        "image8.jpg",
        "image9.jpg",
        "image10.jpg"
    ],

    "Label": [
        0, 0, 0, 1, 1,
        1, 0, 0, 0, 1
    ]
}

df = pd.DataFrame(data)

print("Multimodal Dataset:")
print(df)


# Text feature extraction
vectorizer = TfidfVectorizer()

text_features = vectorizer.fit_transform(df["Text"])


# Image feature extraction
image_features = []

for image_name in df["Image"]:

    try:
        image = Image.open(image_name).convert("RGB")
        image = image.resize((64, 64))

        image_array = np.array(image)

        mean_value = image_array.mean()
        std_value = image_array.std()

        image_features.append([
            mean_value,
            std_value
        ])

    except:
        image_features.append([
            0,
            0
        ])


image_features = np.array(image_features)


# Convert text features to array
text_features_array = text_features.toarray()


# Combine text and image features
combined_features = np.hstack(
    (text_features_array, image_features)
)

print("\nCombined Feature Shape:")
print(combined_features.shape)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    combined_features,
    df["Label"],
    test_size=0.3,
    random_state=42
)


# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# New news prediction
new_news = [
    "Scientists announce a new educational technology"
]

new_text_features = vectorizer.transform(
    new_news
).toarray()

new_image_features = np.array([
    [120, 45]
])

new_combined_features = np.hstack(
    (new_text_features, new_image_features)
)

prediction = model.predict(
    new_combined_features
)

if prediction[0] == 1:
    print("\nPrediction: FAKE NEWS")
else:
    print("\nPrediction: REAL NEWS")