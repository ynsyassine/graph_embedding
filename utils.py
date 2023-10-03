import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

classifiers = [
    ("Logistic Regression", LogisticRegression()),
    ("Random Forest", RandomForestClassifier()),
    ("SVM", SVC()),
    ("k-NN", KNeighborsClassifier()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("XGBoost", XGBClassifier()),
]

def load_graph():
    G = nx.karate_club_graph()
    #G =nx.erdos_renyi_graph(n=10000, p=0.01)
    return G


def graph_embedding(G):
    # Precompute probabilities and generate walks:
    node2vec = Node2Vec(G, dimensions=20, walk_length=16, num_walks=100, workers=4)

    # Embed nodes:
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model


def visualize_graph(G):
    # Visualization:
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1000, font_size=15)
    plt.show()


def generate_positive_edges(G):
    """

    :param G: is the graph thaht we are working with
    :return: positive_egdes : this function return the positive sample as the first list
    :return: all_non_edges : this is a list contain all the non_enges of our graph
    :return: negative_samples :  this list contain the negative_samples
    """
    random.seed(42)
    positive_samples = list(G.edges())
    all_non_edges = list(nx.non_edges(G))
    negative_samples = random.sample(all_non_edges, len(positive_samples))
    return positive_samples , all_non_edges , negative_samples




def create_features(sample, model):
    return np.hstack([model.wv[str(i)] for i in sample])


def X_and_y_data_related_to_Our_graph(model ,positive_samples , negative_samples):
    #creting thee X data for positive and negative
    X_positive = np.array([create_features(sample, model) for sample in positive_samples])
    X_negative = np.array([create_features(sample, model) for sample in negative_samples])
    X = np.vstack([X_positive, X_negative])

    # Labels for supervised learning
    y_positive = np.ones(len(positive_samples))
    y_negative = np.zeros(len(negative_samples))
    y = np.hstack([y_positive, y_negative])

    return X , y


def spliter_of_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train , X_test ,y_train ,  y_test



def scaler(X_train , X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled , X_test_scaled

def classifier_models(classifiers, X_train_scaled , X_test_scaled ,y_train ,  y_test):
    best_score = 0
    best_clf_name = ""

    for name, clf in classifiers:
        clf.fit(X_train_scaled, y_train)
        score = clf.score(X_test_scaled, y_test)
        print(f"{name} Validation Accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_clf_name = name

    print(f"Best Model: {best_clf_name} with Accuracy: {best_score:.4f}")
    return best_clf_name , best_score
