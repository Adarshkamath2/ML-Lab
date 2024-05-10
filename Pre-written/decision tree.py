# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/weather.csv"
df = pd.read_csv(url)

# Data preprocessing
# Convert categorical variables to numerical
df['Outlook'] = df['Outlook'].map({'sunny': 0, 'overcast': 1, 'rainy': 2})
df['Temperature'] = df['Temperature'].map({'hot': 0, 'mild': 1, 'cool': 2})
df['Humidity'] = df['Humidity'].map({'high': 0, 'normal': 1})
df['Wind'] = df['Wind'].map({'weak': 0, 'strong': 1})
df['PlayTennis'] = df['PlayTennis'].map({'no': 0, 'yes': 1})

# Split data into features and target variable
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier()

# Fit the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
