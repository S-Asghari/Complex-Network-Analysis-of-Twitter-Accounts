import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

user_fields = ['followers_count', 'friends_count', 'created_at']
# Add: tweet_count, retweet_count, reply_count, favorite_count, num_hashtags, num_urls, num_mentions

fake_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\FSF\users.csv', usecols=user_fields)
genuine_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\E13\users.csv', usecols=user_fields)

X = fake_user_df.values.tolist() + genuine_user_df.values.tolist()
X = [[x[0], x[1], datetime.strptime(x[2], "%a %b %d %H:%M:%S +0000 %Y").timestamp()] for x in X]
Y = [0 for i in range(len(fake_user_df))] + [1 for i in range(len(genuine_user_df))]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=27)


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
    accuracy_scr = logreg_classifier(X_train, Y_train, X_test, Y_test)
elif classifier == '2':
    accuracy_scr = svm_classifier(X_train, Y_train, X_test, Y_test)
# elif classifier == '3':
#     accuracy = knn_classifier(X_train, Y_train, X_test, Y_test)
# elif classifier == '4':
#     accuracy = rf_classifier(X_train, Y_train, X_test, Y_test)
# elif classifier == '5':
#     accuracy = cnn_classifier(X_train, Y_train, X_test, Y_test)
# elif classifier == '6':
#     accuracy = lstm_classifier(X_train, Y_train, X_test, Y_test)

print("accuracy: " + str(accuracy_scr))
