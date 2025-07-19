#Step 1: Import libraries
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

#Step 2: Load dataset
data = pd.read_csv('E:/4th Year 2nd Semester\Machine Learning Lab\ML Lab Code\Exp-3\play_tennis.csv')

#Step 3: Encode categorical variables
le = preprocessing.LabelEncoder()
for column in data.columns:
    if data[column].dtype == object:
        data[column] = le.fit_transform(data[column])

#Step 4: Train decision tree
X = data.drop(['Play'], axis=1)
y = data['Play']
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X, y)

#Step 5: Visualize
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

#Step 6: Evaluate
accuracy = clf.score(X, y)
print(f"Training accuracy: {accuracy:.2f}")
