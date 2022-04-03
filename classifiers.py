import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

user_fields = ['id', 'followers_count', 'friends_count', 'created_at']
tweet_fields = ['user_id', 'retweet_count', 'favorite_count', 'num_hashtags', 'num_urls']

fake_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\FSF\users.csv', usecols=user_fields)
fake_tweet_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\FSF\tweets.csv', usecols=tweet_fields, encoding='latin1')
genuine_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\E13\users.csv', usecols=user_fields)
genuine_tweet_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\E13\tweets.csv', usecols=tweet_fields, encoding='latin1')

X = fake_user_df.values.tolist() + genuine_user_df.values.tolist()  # X:features
X2 = fake_tweet_df.values.tolist() + genuine_tweet_df.values.tolist()
X = [[x[0], x[1], x[2], datetime.strptime(x[3], "%a %b %d %H:%M:%S +0000 %Y").timestamp(), 0, 0, 0, 0, 0] for x in X]
for x in X:
    for x2 in X2:
        if x2[0] == x[0]:
            x[4] += 1       # tweet count
            x[5] += x2[1]
            x[6] += x2[2]
            x[7] += x2[3]
            x[8] += x2[4]
X = [[x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]] for x in X if all(str(i) != 'nan' for i in x)]
y = [0 for i in range(len(fake_user_df))] + [1 for i in range(len(genuine_user_df))]    # Y:labels
print(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
n_fold = 5
kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)


def logreg_classifier(X_train, y_train, X_test, y_test):

    logreg_clf = LogisticRegression()
    logreg_clf.fit(X=X_train, y=y_train)
    logreg_prediction = logreg_clf.predict(X=X_test)
    return accuracy_score(y_true=y_test, y_pred=logreg_prediction)


def svm_classifier(X_train, y_train, X_test, y_test):

    SVC_model = SVC()
    SVC_model.fit(X=X_train, y=y_train)
    SVC_prediction = SVC_model.predict(X=X_test)
    return accuracy_score(y_true=y_test, y_pred=SVC_prediction)


def knn_classifier(X_train, y_train, X_test, y_test):

    KNN_model = KNeighborsClassifier(n_neighbors=10)
    KNN_model.fit(X=X_train, y=y_train)
    KNN_prediction = KNN_model.predict(X=X_test)
    return accuracy_score(y_true=y_test, y_pred=KNN_prediction)


def rf_classifier(X_train, y_train, X_test, y_test):

    rf_clf = RandomForestClassifier()
    rf_clf.fit(X=X_train, y=y_train)
    rf_prediction = rf_clf.predict(X=X_test)
    return accuracy_score(y_true=y_test, y_pred=rf_prediction)


def cnn_classifier(X_train, y_train, X_test, y_test):

    mlp_clf = MLPClassifier(hidden_layer_sizes=(12, 5), max_iter=300, random_state=1)
    mlp_clf.fit(X=X_train, y=y_train)
    mlp_prediction = mlp_clf.predict(X=X_test)
    return accuracy_score(y_true=y_test, y_pred=mlp_prediction)


# def lstm_classifier(X_train, y_train, X_test, y_test):

print("Which classifier do you want to use?")
print("1. Logistic Regression")
print("2. Support Vector Machine")
print("3. K Nearest Neighbors")
print("4. Random Forest")
print("5. Convolutional Neural Network")
print("6. Long Short-Term Memory Network")
classifier = input()

accuracy_scr = 0

for train_index, test_index in kf.split(X):

    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

    if classifier == '1':
        accuracy_scr += logreg_classifier(X_train, y_train, X_test, y_test)
    elif classifier == '2':
        accuracy_scr += svm_classifier(X_train, y_train, X_test, y_test)
    elif classifier == '3':
        accuracy_scr += knn_classifier(X_train, y_train, X_test, y_test)
    elif classifier == '4':
        accuracy_scr += rf_classifier(X_train, y_train, X_test, y_test)
    elif classifier == '5':
        accuracy_scr += cnn_classifier(X_train, y_train, X_test, y_test)
    # elif classifier == '6':
    #     accuracy_scr += lstm_classifier(X_train, y_train, X_test, y_test)


print("accuracy: " + str(accuracy_scr/n_fold))
