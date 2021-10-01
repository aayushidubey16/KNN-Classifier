import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import classification_report, confusion_matrix


def split_input_data(data):
    X = data[['paramA','paramB']]
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
    return X_train, X_test, y_train, y_test

def display_contours(classifier, number_of_neighbors):
    plot_decision_regions(X_train.values, y_train.values, clf=classifier, legend=2)
    plt.title("Knn with K="+ str(number_of_neighbors))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def knn(nneighbors, X_train, y_train, X_test):
    #Create KNN Classifier
    clf = KNeighborsClassifier(nneighbors,p=2,metric='euclidean')

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    predicted_y = clf.predict(X_test)
    display_contours(clf, nneighbors)
    return predicted_y

def evaluateknn(predicted_y , y_test):
    print(classification_report(y_test, predicted_y))
    return print(confusion_matrix(y_test, predicted_y))

if __name__ == "__main__":
    input_data = pd.read_csv("A1-inputData.csv")
    X_train, X_test, y_train, y_test = split_input_data(input_data)
    predicted_y = knn(3, X_train, y_train, X_test)
    evaluateknn(predicted_y, y_test)
