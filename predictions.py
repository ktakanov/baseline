__author__ = 'pdanilov'

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.cross_validation import train_test_split


def fit_data(features, outputs):
    classifier = RandomForestClassifier(n_estimators=20, min_samples_split=150, class_weight='auto', n_jobs=-1)
    classifier.fit(features, outputs)
    return classifier


def predict_buy_or_not(classifier, buy_or_not_features):
    return classifier.predict(buy_or_not_features)


def metrics(y_true, y_pred):
    precision, recall, f1_score = precision_recall_fscore_support(y_true, y_pred, average='binary')[0:3]
    accuracy = accuracy_score(y_true, y_pred)
    return [precision, recall, f1_score, accuracy]

def split_data(x, y, test_size):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size)
    # return [x_train, x_test, np.array(y_train['Prediction'], dtype=np.intc), np.array(y_test['Prediction'], dtype=np.intc)]
    return [x_train, x_val, np.ravel(y_train), np.ravel(y_val)]
