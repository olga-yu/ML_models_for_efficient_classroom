import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Load your dataset
columns = ['sensor_mo.mean', 'Month', 'Year', 'WeekDay', 'TimeOfDay', 'Semester']
data = pd.read_csv(r'../data/output_14_18_7.csv', header=None, names=columns, skipinitialspace=True, skiprows=1)
# Drop rows with missing data (if any)
data.dropna(inplace=True)

# Prepare features and target
X = data.drop("sensor_mo.mean", axis=1)  # Features
y = data['sensor_mo.mean']  # Target variable
print(X)
# # Encode categorical features as one-hot encodings
X = pd.get_dummies(X, columns=['Month','Year', 'WeekDay','TimeOfDay', 'Semester'], drop_first=True)
#
# # Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Create and fit the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
#
# # Predict the target variable for the test set
y_pred = clf.predict(X_test)
#
# # Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
#
# # Print results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# importances = clf.feature_importances_
# indices = np.argsort(importances)[::-1]
#
# plt.figure()
# plt.title("Feature Importances")
# plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
# plt.xticks(range(X.shape[1]), [pd[i] for i in indices], rotation=90)
# plt.xlim([-1, X.shape[1]])
# plt.show()
