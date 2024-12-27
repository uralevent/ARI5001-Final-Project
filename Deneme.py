import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
path = r"titanic.csv"
titanic_data = pd.read_csv(path)

# Inspect the dataset
print(titanic_data.head())
print(titanic_data.info())

# Check and handle missing data
print(titanic_data.isnull().sum())

# Fill missing values in the 'Age' column with the median
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Fill missing values in the 'Embarked' column with the mode
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numerical values
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Define independent and dependent variables
X = titanic_data.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
y = titanic_data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
y_pred = rf_classifier.predict(X_test_scaled)

# Assess model performance
print("Random Forest Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Visualize the confusion matrix for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Create and train the Decision Tree model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
y_pred_dt = dt_classifier.predict(X_test_scaled)

# Assess model performance
print("Decision Tree Accuracy Score:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt, zero_division=0))

# Visualize the confusion matrix for Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# Exploratory data visualizations
# Survival rates by gender
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_data, x='Sex', hue='Survived')
plt.title('Survival Rates by Gender')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Number of People')
plt.legend(['Did Not Survive', 'Survived'])
plt.show()

# Age distribution on the ship
plt.figure(figsize=(10, 6))
sns.histplot(titanic_data['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.show()

# Survival rates by ticket class
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_data, x='Pclass', hue='Survived')
plt.title('Survival Rates by Class')
plt.xlabel('Ticket Class')
plt.ylabel('Number of People')
plt.legend(['Did Not Survive', 'Survived'])
plt.show()

# Survival rates by port of embarkation
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_data, x='Embarked', hue='Survived')
plt.title('Survival Rates by Port')
plt.xlabel('Boarding from the Port (0 = Southampton, 1 = Cherbourg, 2 = Queenstown)')
plt.ylabel('Number of People')
plt.legend(['Did Not Survive', 'Survived'])
plt.show()

# Relationship between age and survival
plt.figure(figsize=(10, 6))
sns.boxplot(data=titanic_data, x='Survived', y='Age')
plt.title('Relationship Between Age and Survival')
plt.xlabel('Survival (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()
