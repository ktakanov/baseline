__author__ = 'pdanilov'

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.cross_validation import train_test_split
import numpy as np


def fit_data(features, outputs_df):
    classifier = RandomForestClassifier(min_samples_split=200, njobs=-1)
    outputs = outputs_df['Prediction']
    classifier.fit(features, outputs)
    return classifier


def predict_buy_or_not(classifier, buy_or_not_features):
    return classifier.predict(buy_or_not_features)


def metrics(y_true, y_pred):
    precision, recall = precision_recall_fscore_support(y_true, y_pred, average='binary')[0:2]
    accuracy = accuracy_score(y_true, y_pred)
    return [precision, recall, accuracy]

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return [X_train, X_test, y_train, y_test]
