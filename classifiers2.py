import pandas as pd
import numpy as np
import networkx as nx
from itertools import chain
from statistics import mean
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

user_fields = ['id', 'followers_count', 'friends_count', 'created_at']
tweet_fields = ['user_id', 'retweet_count', 'favorite_count', 'num_hashtags', 'num_urls']
fake_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\social_spambots_2\users.csv', usecols=user_fields)
fake_tweet_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\social_spambots_2\tweets.csv', usecols=tweet_fields, encoding='latin1')
genuine_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\genuine_accounts\users.csv', usecols=user_fields)
genuine_tweet_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\genuine_accounts\tweets.csv', usecols=tweet_fields, encoding='latin1')
G = nx.read_gexf(r'C:\Users\Sara\Desktop\twitterProject\results\storage\commentGraph.gexf')
G2 = nx.read_gexf(r'C:\Users\Sara\Desktop\twitterProject\results\storage\commentGraph2.gexf')


def Average(lst):

    avg = 0
    if len(lst) > 0:
        avg = mean(lst)
    return avg


shortest_path_len = dict(nx.shortest_path_length(G2))
avg_incoming_path_len = {v: Average([shortest_path_len[u][v] for u in G2.nodes if u != v and nx.has_path(G2,u,v)]) for v in G2.nodes}
avg_outgoing_path_len = {u: Average([shortest_path_len[u][v] for v in G2.nodes if v != u and nx.has_path(G2,u,v)]) for u in G2.nodes}
avg_path_len = {n: mean([avg_incoming_path_len[n], avg_outgoing_path_len[n]])
                    if (avg_incoming_path_len[n] > 0 and avg_outgoing_path_len[n] > 0)
                    else max(avg_incoming_path_len[n], avg_outgoing_path_len[n]) for n in G2.nodes}
avg_neighbor_degree = nx.average_neighbor_degree(G, weight='weight')

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
X_prime = [[x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8],
            avg_path_len[str(x[0])] if str(x[0]) in G2.nodes else 0.0,
            avg_neighbor_degree[str(x[0])] if str(x[0]) in G.nodes else 0.0]
           for x in X if all(str(i) != 'nan' for i in x)]
X = [[x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]] for x in X if all(str(i) != 'nan' for i in x)]
y = [0 for i in range(len(fake_user_df))] + [1 for i in range(len(genuine_user_df))]    # Y:labels

X_mean = [mean([x[i] for x in X]) for i in range(8)]
X_max = [max([abs(x[i]) for x in X]) for i in range(8)]
X = [[(x[i]-X_mean[i])/X_max[i] for i in range(3, 5)] for x in X]
X_prime_mean = [mean([x_p[i] for x_p in X_prime]) for i in range(10)]
X_prime_max = [max([abs(x_p[i]) for x_p in X_prime]) for i in range(10)]
X_prime = [[(x_p[i]-X_prime_mean[i])/X_prime_max[i] for i in chain(range(3, 5), range(8, 10))] for x_p in X_prime]


def logreg_classifier(X_train, y_train, X_test, y_test):

    logreg_clf = LogisticRegression()
    logreg_clf.fit(X=X_train, y=y_train)
    logreg_prediction = logreg_clf.predict(X=X_test)
    return accuracy_score(y_true=y_test, y_pred=logreg_prediction), f1_score(y_true=y_test, y_pred=logreg_prediction)


def svm_classifier(X_train, y_train, X_test, y_test):

    SVC_model = SVC()
    SVC_model.fit(X=X_train, y=y_train)
    SVC_prediction = SVC_model.predict(X=X_test)
    return accuracy_score(y_true=y_test, y_pred=SVC_prediction), f1_score(y_true=y_test, y_pred=SVC_prediction)


def knn_classifier(X_train, y_train, X_test, y_test):

    KNN_model = KNeighborsClassifier(n_neighbors=10)
    KNN_model.fit(X=X_train, y=y_train)
    KNN_prediction = KNN_model.predict(X=X_test)
    return accuracy_score(y_true=y_test, y_pred=KNN_prediction), f1_score(y_true=y_test, y_pred=KNN_prediction)


def rf_classifier(X_train, y_train, X_test, y_test):

    rf_clf = RandomForestClassifier()
    rf_clf.fit(X=X_train, y=y_train)
    rf_prediction = rf_clf.predict(X=X_test)
    return accuracy_score(y_true=y_test, y_pred=rf_prediction), f1_score(y_true=y_test, y_pred=rf_prediction)


def cnn_classifier(X_train, y_train, X_test, y_test):

    mlp_clf = MLPClassifier(hidden_layer_sizes=(12, 5), max_iter=300, random_state=1)
    mlp_clf.fit(X=X_train, y=y_train)
    mlp_prediction = mlp_clf.predict(X=X_test)
    return accuracy_score(y_true=y_test, y_pred=mlp_prediction), f1_score(y_true=y_test, y_pred=mlp_prediction)


# def lstm_classifier(X_train, y_train, X_test, y_test):


def evaluate_classifiers(X, y):

    accuracy_scr = [[0 for j in range(3)] for i in range(5)]
    f1_scr = [[0 for j in range(3)] for i in range(5)]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
    for n_fold in range(3, 6):
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
            accuracy_scr[0][n_fold - 3] += logreg_classifier(X_train, y_train, X_test, y_test)[0]
            accuracy_scr[1][n_fold - 3] += svm_classifier(X_train, y_train, X_test, y_test)[0]
            accuracy_scr[2][n_fold - 3] += knn_classifier(X_train, y_train, X_test, y_test)[0]
            accuracy_scr[3][n_fold - 3] += rf_classifier(X_train, y_train, X_test, y_test)[0]
            accuracy_scr[4][n_fold - 3] += cnn_classifier(X_train, y_train, X_test, y_test)[0]
            # accuracy_scr[5][n_fold-3] += lstm_classifier(X_train, y_train, X_test, y_test)
            f1_scr[0][n_fold - 3] += logreg_classifier(X_train, y_train, X_test, y_test)[1]
            f1_scr[1][n_fold - 3] += svm_classifier(X_train, y_train, X_test, y_test)[1]
            f1_scr[2][n_fold - 3] += knn_classifier(X_train, y_train, X_test, y_test)[1]
            f1_scr[3][n_fold - 3] += rf_classifier(X_train, y_train, X_test, y_test)[1]
            f1_scr[4][n_fold - 3] += cnn_classifier(X_train, y_train, X_test, y_test)[1]
    return accuracy_scr, f1_scr


def print_results(accuracy_scr, f1_scr):

    print("1. Logistic Regression ➤ AVG of accuracy: " + str(sum(accuracy_scr[0][n_fold-3]/n_fold for n_fold in range(3, 6)) / 3) +
          " | AVG of F1-score: " + str(sum(f1_scr[0][n_fold-3]/n_fold for n_fold in range(3, 6)) / 3))
    print("2. Support Vector Machine ➤ accuracy: " + str(sum(accuracy_scr[1][n_fold-3]/n_fold for n_fold in range(3, 6)) / 3) +
          " | AVG of F1-score: " + str(sum(f1_scr[1][n_fold - 3] / n_fold for n_fold in range(3, 6)) / 3))
    print("3. K Nearest Neighbors ➤ accuracy: " + str(sum(accuracy_scr[2][n_fold-3]/n_fold for n_fold in range(3, 6)) / 3) +
          " | AVG of F1-score: " + str(sum(f1_scr[2][n_fold - 3] / n_fold for n_fold in range(3, 6)) / 3))
    print("4. Random Forest ➤ accuracy: " + str(sum(accuracy_scr[3][n_fold-3]/n_fold for n_fold in range(3, 6)) / 3) +
          " | AVG of F1-score: " + str(sum(f1_scr[3][n_fold - 3] / n_fold for n_fold in range(3, 6)) / 3))
    print("5. Convolutional Neural Network ➤ accuracy: " + str(sum(accuracy_scr[4][n_fold-3]/n_fold for n_fold in range(3, 6)) / 3) +
          " | AVG of F1-score: " + str(sum(f1_scr[4][n_fold-3]/n_fold for n_fold in range(3, 6)) / 3))
    # print("6. Long Short-Term Memory Network ➤ accuracy: " + str(sum(accuracy_scr[5][n_fold-3]/n_fold for n_fold in range(3, 6)) / 3))


accuracy_scr, f1_scr = evaluate_classifiers(X, y)
print("Without using the centrality measures:")
print_results(accuracy_scr, f1_scr)

accuracy_scr, f1_scr = evaluate_classifiers(X_prime, y)
print("\nUsing the centrality measures:")
print_results(accuracy_scr, f1_scr)
