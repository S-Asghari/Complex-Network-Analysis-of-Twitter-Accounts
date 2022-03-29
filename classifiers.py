import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

user_fields = ['id']    # must be changed!
fake_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\FSF\users.csv', usecols=user_fields)
genuine_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\E13\users.csv', usecols=user_fields)


def logreg_classifier(X_train, Y_train, X_test, Y_test):

    logreg_clf = LogisticRegression()
    logreg_clf.fit(X_train, Y_train)  # X:features, Y:labels
    logreg_prediction = logreg_clf.predict(X_test)
    return accuracy_score(logreg_prediction, Y_test)


def svm_classifier(X_train, Y_train, X_test, Y_test):

    SVC_model = SVC()
    SVC_model.fit(X_train, Y_train)
    SVC_prediction = SVC_model.predict(X_test)
    return accuracy_score(SVC_prediction, Y_test)


# def knn_classifier(X_train, Y_train, X_test, Y_test):
#
#
# def rf_classifier(X_train, Y_train, X_test, Y_test):
#
#
# def cnn_classifier(X_train, Y_train, X_test, Y_test):
#
#
# def lstm_classifier(X_train, Y_train, X_test, Y_test):

print("Which classifier do you want to use?")
print("1. Logistic Regression")
print("2. Support Vector Machine")
print("3. K Nearest Neighbors")
print("4. Random Forest")
print("5. Convolutional Neural Network")
print("6. Long Short-Term Memory Network")
classifier = input()

if classifier == '1':
    accuracy = logreg_classifier(X_train, Y_train, X_test, Y_test)
elif classifier == '2':
    accuracy = svm_classifier(X_train, Y_train, X_test, Y_test)
# elif classifier == '3':
#     accuracy = knn_classifier(X_train, Y_train, X_test, Y_test)
# elif classifier == '4':
#     accuracy = rf_classifier(X_train, Y_train, X_test, Y_test)
# elif classifier == '5':
#     accuracy = cnn_classifier(X_train, Y_train, X_test, Y_test)
# elif classifier == '6':
#     accuracy = lstm_classifier(X_train, Y_train, X_test, Y_test)

print("accuracy: " + str(accuracy))