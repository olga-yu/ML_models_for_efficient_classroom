import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
columns = ['Time', 'sensor_mo.mean', 'Month', 'Year', 'Extracted_Time', 'WeekDay', 'TimeOfDay', 'Semester', 'User_sensor_mo.mean']
data = pd.read_csv(r'../data/output_9_18_7.csv', header=None, names=columns, skipinitialspace=True, skiprows=1)

# Drop rows with missing values
data.dropna(inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop("sensor_mo.mean", axis=1)
y = data['sensor_mo.mean']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualize the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=["0", "1"])
plt.show()
