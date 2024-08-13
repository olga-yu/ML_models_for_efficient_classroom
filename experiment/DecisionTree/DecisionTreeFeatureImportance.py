from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import pandas as pd

# define dataset
X, y = make_classification(n_samples=1000, n_features=11, n_informative=5, n_redundant=6, random_state=1)

dataset = pd.read_csv(r'../data/output_14_18_7.csv')

# Select the relevant features
feature_cols = ['Month', 'Year', 'WeekDay', 'TimeOfDay', 'Semester']

# Extract the features and target variable
X = dataset[feature_cols]
y = dataset['sensor_mo.mean']

# define the model

model = DecisionTreeClassifier()
# fit the model
model.fit(X, y)
# get importance

importance = model.feature_importances_
# summarize feature importance

for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))

for feature, score in zip(feature_cols, importance):
    print(feature, score)

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.figure(figsize=(10, 6))  # Adjust the figure size as needed
pyplot.bar(range(len(importance)), importance, tick_label=feature_cols)
pyplot.xlabel('Features')
pyplot.ylabel('Importance Score')
pyplot.title('5 Features Importance Scores')
pyplot.xticks(rotation=45)  # Rotate x-axis labels for better visibility if needed
pyplot.tight_layout()

pyplot.show()
