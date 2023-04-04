from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import pickle
import numpy as np
import preprocessor


def predict_load_model(sample_text, model_path):
    word_list = preprocessor.word_extraction(sample_text)
    text = ' '.join(word_list)

    model = SentenceTransformer('sentence-transformers/LaBSE')

    vector = model.encode(text)
    xx = [vector]
    print("Loading model: ", model_path)
    loaded_model = pickle.load(open(model_path, 'rb'))
    y_pred = loaded_model.predict(xx)
    print(y_pred)
    print("Sample text: ", sample_text, "\nClassified as " + ("ham" if y_pred[0] == 0 else "spam"))


# Gaussian Naive Bayes training.
def train_gnb(X, y):
    # Split and train
    print("Training Gaussian Naive Bayes.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print("Complete!")

    # Save model checkpoint
    filename = 'models/gNB.sav'
    pickle.dump(gnb, open(filename, 'wb'))
    print("Saved model checkpoint as gNB.sav")

    # Evaluate model
    y_pred = gnb.predict(X_test)
    acc = np.mean(y_pred == y_test)

    y = np.array(y_test, dtype=float)
    pred = np.array(y_pred, dtype=float)
    score = roc_auc_score(y, pred)

    print("Accuracy: ", acc)
    print(metrics.classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    print(f"ROC AUC: {score:.4f}\n")

    return


def train_rf(X, y):
    # Split and train
    print("Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    rf = RandomForestClassifier(max_depth=10, random_state=0)
    rf.fit(X, y)
    print("Complete!")

    # Save model checkpoint
    filename = 'models/RF.sav'
    pickle.dump(rf, open(filename, 'wb'))
    print("Saved model checkpoint as ", filename)

    # Evaluate model
    y_pred = rf.predict(X_test)
    acc = np.mean(y_pred == y_test)

    y = np.array(y_test, dtype=float)
    pred = np.array(y_pred, dtype=float)
    score = roc_auc_score(y, pred)

    print("Accuracy: ", acc)
    print(metrics.classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    print(f"ROC AUC: {score:.4f}\n")


def train_knn(X, y):
    # Split and train
    print("Training k-Nearest Neighbors...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    print("Complete!")

    # Save model checkpoint
    filename = 'models/kNN.sav'
    pickle.dump(neigh, open(filename, 'wb'))
    print("Saved model checkpoint as ", filename)

    # Evaluate model
    y_pred = neigh.predict(X_test)
    acc = np.mean(y_pred == y_test)

    y = np.array(y_test, dtype=float)
    pred = np.array(y_pred, dtype=float)
    score = roc_auc_score(y, pred)

    print("Accuracy: ", acc)
    print(metrics.classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    print(f"ROC AUC: {score:.4f}\n")


def classifiers_main(dataset_path):
    X, y = preprocessor.open_dataset(dataset_path)
    preprocessor.print_prior_stats(X, y)

    train_gnb(X, y)
    train_knn(X, y)
    train_rf(X, y)
    return
