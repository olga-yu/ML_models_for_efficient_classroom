import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
#columns = ['Time', 'sensor_mo.mean', 'Month', 'Year', 'Extracted_Time', 'WeekDay', 'TimeOfDay', 'Semester', 'User_sensor_mo.mean']
columns = ['Time', 'sensor_mo.mean', 'Month', 'Year', 'Extracted_Time', 'WeekDay', 'TimeOfDay', 'Semester']
#data = pd.read_csv(r'../data/output_9_18_7.csv', header=None, names=columns, skipinitialspace=True, skiprows=1)
#data = pd.read_csv(r'../data/output_9_13_7.csv', header=None, names=columns, skipinitialspace=True, skiprows=1)
data = pd.read_csv(r'../data/output_14_18_7.csv', header=None, names=columns, skipinitialspace=True, skiprows=1)


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
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Visualize the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=["0", "1"])
plt.show()
