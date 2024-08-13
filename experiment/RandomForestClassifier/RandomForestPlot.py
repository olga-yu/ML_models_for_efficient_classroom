import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
columns = ['sensor_mo.mean', 'Month', 'Year', 'WeekDay', 'TimeOfDay', 'Semester']
data = pd.read_csv(r'../data/output_14_18_7.csv', header=None, names=columns, skipinitialspace=True, skiprows=1)

# Drop rows with missing data (if any)
data.dropna(inplace=True)

# Prepare features and target
X = data.drop("sensor_mo.mean", axis=1)  # Features
y = data['sensor_mo.mean']  # Target variable

# Encode categorical features as one-hot encodings
X = pd.get_dummies(X, columns=['Month', 'Year', 'WeekDay', 'TimeOfDay', 'Semester'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Define the function for plotting feature importance
def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order of decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))

    # Plot Seaborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

    # Display the plot
    plt.show()

# Call the function to plot feature importance
plot_feature_importance(clf.feature_importances_, X.columns, 'Random Forest')
