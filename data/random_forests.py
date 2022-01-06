import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from utilities import visualize_classifier



#Argument parser
def build_arg_parser():
    parser = argparse.ArgumentParser(description = 'Classify data using Ensemble learning techniques')
    parser.add_argument('--classifier-type', dest = 'classifier_type', required = True, choices = ['rf', 'erf'],
                        help = "Type of classifier to use, either 'rf' or 'erf'")
                        
    return parser
                        

if __name__ == '__main__':
    #parse the input arguments
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type
    
    #Load input data
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter = ',')
    X, y = data[:, :-1], data[:, -1]
    
    #Separate input data into classes based on labels (3 in this case)
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])
    
    #Visualize the input data
    plt.figure()
    plt.title('Input data')
    
    plt.scatter(class_0[:, 0], class_0[:, 1], s = 75, facecolors = 'white', edgecolors = 'black', linewidth = 1, marker = 's')
    plt.scatter(class_1[:, 0], class_1[:, 1], s = 75, facecolors = 'white', edgecolors = 'black', linewidth = 1, marker = 'o')
    plt.scatter(class_2[:, 0], class_2[:, 1], s = 75, facecolors = 'white', edgecolors = 'black', linewidth = 1, marker = '^')    
    
    #Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)
    
    #Ensemble learning classifier
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)
    else:
        classifier = ExtraTreesClassifier(**params)
        
    classifier.fit(X_train, y_train)
    
    visualize_classifier(classifier, X_train, y_train, 'Training dataset')
    visualize_classifier(classifier, X_test, y_test, 'Test dataset')
    
    y_test_pred = classifier.predict(X_test)
    
    #Evaluate classifier performance
    class_names = ['Class-0', 'Class-1', 'Class-2']
    
    print('#' * 40)
    print('\nClassifier performance on the training dataset\n')
    
    y_train_pred = classifier.predict(X_train)
    print(classification_report(y_train, y_train_pred, target_names = class_names), '\n\n')
    
    print('#' * 40)
    print('\nClassifier performance on the test dataset\n')
    print(classification_report(y_test, y_test_pred, target_names = class_names))
    
    #Compute confidence
    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])
    
    print('\nConfidence measure: ')
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = f'Class-{np.argmax(probabilities)}'
        
        print(f'\nDatapoint: {datapoint}')
        print(f'Probabilites: {probabilities}')
        print(f'predicted class: {predicted_class}')
        
    #Visualize the datapoints
    visualize_classifier(classifier, test_datapoints, [0] * len(test_datapoints), 'Test Datapoints')
    plt.show()