from read_and_write_data import read_clicks, read_buys, write_predictions, write_metrics,\
    features_to_csv, features_from_csv
from feature_extraction import extract_buy_or_not, extract_what_to_buy, extract_buys
from predictions import metrics
from preprocess_data import df_group_by_session_id
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


if __name__ == '__main__':
    assert len(sys.argv) == 9, ('Incorrect argument list, names also shouldn\'t contain spaces' + '\n' + 'usage:'
                                'argv[0] /working/dir clicks_file buys_file test_file what_to_buy_file buy_or_not_file'
                                'scores_file resulting_file')

    path = sys.argv[1]
    file_clicks_basename = sys.argv[2]
    file_buys_basename = sys.argv[3]
    file_test_basename = sys.argv[4]
    file_what_to_buy_features_basename = sys.argv[5]
    file_buy_or_not_features_basename = sys.argv[6]
    file_scores_basename = sys.argv[7]
    file_result_basename = sys.argv[8]

    data_path = os.path.join(path, 'data')
    cache_path = os.path.join(path, 'cache')
    prediction_path = os.path.join(path, 'prediction')
    file_clicks = os.path.join(data_path, file_clicks_basename)
    file_buys = os.path.join(data_path, file_buys_basename)
    file_test = os.path.join(data_path, file_test_basename)
    file_what_to_buy_features = os.path.join(cache_path, file_what_to_buy_features_basename)
    file_buy_or_not_features = os.path.join(cache_path, file_buy_or_not_features_basename)
    file_scores = os.path.join(prediction_path, file_scores_basename)
    file_result = os.path.join(prediction_path, file_result_basename)

    effective_columns_names = ['Session ID', 'Timestamp', 'Item ID']
    clicks = read_clicks(file_clicks, usecols=effective_columns_names)\
        .sort_values('Timestamp')\
        .reset_index(drop=True)
    clicks_grouped_by_session_id, clicks_grouped_by_session_id_keys = df_group_by_session_id(clicks)

    if os.path.isfile(file_what_to_buy_features):
        what_to_buy = features_from_csv(file_what_to_buy_features, ['Session ID', 'Item ID'],
                                        {'F3': 'Counts', 'F6': 'Sequent Clicks', 'F7': 'Time Difference'})
    else:
        what_to_buy = extract_what_to_buy(clicks_grouped_by_session_id)
        features_to_csv(file_what_to_buy_features, list(what_to_buy.values()))

    if os.path.isfile(file_buy_or_not_features):
        buy_or_not = np.load(file_buy_or_not_features)
    else:
        buy_or_not = extract_buy_or_not(clicks_grouped_by_session_id, what_to_buy)
        np.save(file_buy_or_not_features, buy_or_not)

    buys = read_buys(file_buys, usecols=effective_columns_names)
    _, buys_grouped_by_session_id_keys = df_group_by_session_id(buys)
    buys_result = extract_buys(clicks_grouped_by_session_id_keys, buys_grouped_by_session_id_keys)

    buy_or_not_train, buy_or_not_val, buys_result_train, buys_result_val = train_test_split(buy_or_not, buys_result,
                                                                                            test_size=0.2)

    classifier = XGBClassifier(n_estimators=500, subsample=0.8, colsample_bytree=0.5, max_depth=4, min_child_weight=3)
    classifier.fit(buy_or_not_train, buys_result_train)

    predictions_val = classifier.predict(buy_or_not_val)
    scores = metrics(buys_result_val, predictions_val)
    write_metrics(file_scores, scores)

    test = read_clicks(file_test, usecols=effective_columns_names)
    test_grouped_by_session_id, _ = df_group_by_session_id(test)

    what_to_buy_test = extract_what_to_buy(test_grouped_by_session_id)
    buy_or_not_test = extract_buy_or_not(test_grouped_by_session_id, what_to_buy_test)

    predictions_test = classifier.predict(buy_or_not_test)
    write_predictions(file_result, predictions_test)
