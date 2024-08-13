import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
#dataset = pd.read_csv(r'../data/output_9_13_7.csv')
#dataset = pd.read_csv(r'../data/output_14_18_7.csv')
dataset = pd.read_csv(r'../data/output_9_18_7.csv')
#dataset = pd.read_csv(r'../data/output_9_18_5.csv')
# Select the relevant features
#feature_cols = ['sensor_hu.mean','sensor_te.mean','Month', 'Year', 'WeekDay', 'TimeOfDay', 'Semester']
feature_cols = ['Month', 'Year', 'WeekDay', 'TimeOfDay', 'Semester']

# Extract the features and target variable
X = dataset[feature_cols]
y = dataset['sensor_mo.mean']
#y = dataset['test_attendance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Fit the decision tree regression model
regr = DecisionTreeRegressor(max_depth=3)
regr.fit(X_train, y_train)

# Generate predictions for the test set
y_pred = regr.predict(X_test)

# Plot the predicted attendance against the actual attendance
plt.scatter(y_test, y_pred)
plt.xlabel('Actual attendance')
plt.ylabel('Predicted attendance')
plt.title('Decision Tree MLR: 5 features')
#plt.title('Decision Tree MLR: 7 features')
plt.show()

meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r_squared = regr.score(X, y)

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(r_squared * 100))
print('Mean Square Error:', meanSqErr)

# Plot the Residuals
plt.scatter(y_pred, y_test - y_pred)
plt.xlabel('Predicted attendance')
plt.ylabel('Residuals')
plt.title('Residual plot 5 features')
#plt.title('Residual plot 7 features')
plt.show()
