import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utilities import visualize_classifier



#Load input data
input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter = ',')
X, y = data[:, :-1], data[:, -1]

#Separate the input data into classes based on labels (2 in this case)
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

#Visualize input data
plt.figure()
plt.title('Input data')
plt.scatter(class_0[:, 0], class_0[:, 1], s = 75, facecolors = 'black', edgecolors = 'black', linewidth = 1, marker = 'x')
plt.scatter(class_1[:, 0], class_1[:, 1], s = 75, facecolors = 'white', edgecolors = 'black', linewidth = 1, marker = 'o')

#Split the data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)

#Extremely Random forests classifier
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

if len(sys.argv) > 1:
    if sys.argv[1] == 'balance':
        params['class_weight'] = 'balanced'
    else:
        raise TypeError("Invalid input argument, should be 'balance'")
        
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)

visualize_classifier(classifier, X_train, y_train, 'Training dataset')
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

y_test_pred = classifier.predict(X_test)

#Evaluate classifier performance
class_names = ['Class-0', 'Class-1']

print('#' * 40)
print('Classifier performance on training dataset: \n')

y_train_pred = classifier.predict(X_train)
print(classification_report(y_train, y_train_pred, target_names = class_names), '\n')

print('#' * 40)
print('Classifier performance on test dataset: \n')
print(classification_report(y_test, y_test_pred, target_names = class_names, zero_division = 0))
plt.show()