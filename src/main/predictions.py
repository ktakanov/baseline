from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def metrics(y_true, y_pred):
    precision, recall, f1_score = precision_recall_fscore_support(y_true, y_pred, average='binary')[0:3]
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1_score, accuracy
