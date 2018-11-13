import numpy as np
import pandas as pd
import re
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate
import graphviz
from subprocess import check_call
from PIL import Image, ImageDraw

# Original Dataset: Titanic: Machine Learning from Disaster from Kaggle
# Use the 'absolutely right rules' to revise the labels
train_file ='../dataset/Titanic/train.csv'
train = pd.read_csv(train_file)
test_label_file = '../dataset/Titanic/gender_submission.csv'
test_label = pd.read_csv(test_label_file)
test_file = '../dataset/Titanic/test.csv'
test = pd.read_csv(test_file)

PassengerId = test['PassengerId']
original_train = train.copy()
full_data = [train, test]

# Data Preprocessing ...
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
# Remove all NULLS in the Age column
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

# extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search: return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age']                            = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    # Mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Feature selection: remove variables no longer containing relevant information
drop_elements = ['PassengerId', 'Name', 'SibSp', 'Parch']
# Columns : Survived, Pclass, Sex, Age, Fare, Embarked, FamilySize, IsAlone, Title
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)
y_train = train['Survived']
x_train = train.drop(['Survived'], axis=1).values
x_test = test.values
#########################################
###   Decision Tree
#########################################
cv = KFold(n_splits=10)
accuracies = list()
max_attributes = len(list(test))
depth_range = range(1, max_attributes+1)
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
    for train_fold, valid_fold in cv.split(train):
        f_train = train.loc[train_fold] # Extract train data with cv indices
        f_valid = train.loc[valid_fold] # Extract valid data with cv indices
        
        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1),
        y = f_train["Survived"]) # fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1),
        y = f_valid["Survived"])# calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)
    accuracies.append(sum(fold_accuracy)/len(fold_accuracy))

# show results
df = pd.DataFrame({"Max_depth": depth_range, "Avg Accuracy": accuracies})
df = df[["Max_depth", "Avg Accuracy"]]
print('#########################################')
print('Use Cross Validation to test max_depths:')
print(' ',df.to_string(index=False))
# use max_depth = 8 because the accuracy is the highest of all
max_depth = 8
# Create Decision Tree
decision_tree = tree.DecisionTreeClassifier(max_depth = max_depth)
decision_tree.fit(x_train, y_train)

# Predicting results for test dataset
y_pred = decision_tree.predict(x_test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived_predict": y_pred })
submission.to_csv('../results/submission_decisionTree.csv', index=False)
# Export our trained model as a .dot file
with open("../results/tree.dot", 'w') as f:
    f = tree.export_graphviz(decision_tree, out_file=f, max_depth = max_depth,
        impurity = True, feature_names = list(train.drop(['Survived'], axis=1)),
        class_names = ['Died', 'Survived'], rounded = True, filled= True )

#Convert .dot to .png
check_call(['dot','-Tpng','../results/tree.dot','-o','../results/tree.png'])
#########################################
# Comparison of test_label and predicted labels
comp = pd.concat([test_label, submission.iloc[:, 1]], axis=1, sort=False)
print('#########################################')
#print('>>  Decision Tree :')
#print(comp.to_string(index=False))
#print('#########################################')
correct_count = comp[comp.Survived == comp.Survived_predict].count().values[0]
print('Accuracy(Decision Tree): ',correct_count/len(comp))

##########################################
### SVM
#########################################
svm = SVC(kernel='rbf')
print('#########################################')
print(svm)
svm = svm.fit(x_train, y_train)
# Predicting results for test dataset
y_pred_svm = svm.predict(x_test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived_predict": y_pred_svm })
submission.to_csv('../results/submission_svm.csv', index=False)
#########################################
print('\nSVM Cross Validation:')
scores = cross_validate(svm, train.drop(['Survived'], axis=1), y_train, cv=10, scoring='accuracy')
for item in scores.items():
    print('{}:\t{}'.format(item[0], item[1]))

# Comparison of test_label and predicted labels
comp = pd.concat([test_label, submission.iloc[:, 1]], axis=1, sort=False)
print('#########################################')
#print('>>  SVM :')
#print(comp.to_string(index=False))
#print('#########################################')
correct_count = comp[comp.Survived == comp.Survived_predict].count().values[0]
print('Accuracy(SVM): ',correct_count/len(comp))
print('#########################################')

##########################################
### NN
#########################################
# multi-layer perceptron (MLP) algorithm (training using Backpropagation)
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1)
print('#########################################')
print('NN: ',nn)
nn = nn.fit(x_train, y_train)
y_pred_nn = nn.predict(x_test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived_predict": y_pred_nn })
submission.to_csv('../results/submission_nn.csv', index=False)
#########################################
print('\nNN Cross Validation:')
scores = cross_validate(nn, train.drop(['Survived'], axis=1), y_train, cv=10, scoring='accuracy')
for item in scores.items():
    print('{}:\t{}'.format(item[0], item[1]))

# Comparison of test_label and predicted labels
comp = pd.concat([test_label, submission.iloc[:, 1]], axis=1, sort=False)
print('#########################################')
correct_count = comp[comp.Survived == comp.Survived_predict].count().values[0]
print('Accuracy(NN): ',correct_count/len(comp))
print('#########################################')

