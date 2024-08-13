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

# Define the function for plotting aggregated feature importance
def plot_aggregated_feature_importance(importance, feature_names, original_features):
    # Create a dictionary to hold the aggregated importance
    aggregated_importance = {feature: 0 for feature in original_features}

    # Loop through the one-hot encoded feature names and their importance
    for name, imp in zip(feature_names, importance):
        # Find the original feature name in the one-hot encoded name
        for original_feature in original_features:
            if name.startswith(original_feature):
                aggregated_importance[original_feature] += imp
                break

    # Convert the dictionary to a DataFrame for plotting
    fi_df = pd.DataFrame({
        'feature_names': list(aggregated_importance.keys()),
        'feature_importance': list(aggregated_importance.values())
    })

    # Sort the DataFrame by feature importance
    fi_df.sort_values(by='feature_importance', ascending=False, inplace=True)

    # Plot the aggregated feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='feature_importance', y='feature_names', data=fi_df)
    plt.title('Aggregated Feature Importance')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

# List of original features
original_features = ['Month', 'Year', 'WeekDay', 'TimeOfDay', 'Semester']

# Call the function to plot aggregated feature importance
plot_aggregated_feature_importance(clf.feature_importances_, X.columns, original_features)
