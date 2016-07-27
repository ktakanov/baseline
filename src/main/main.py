from read_and_write_data import read_clicks, read_buys, write_predictions, write_metrics
from feature_extraction import extract_buy_or_not, extract_what_to_buy, extract_buys
from predictions import fit_data, predict_buy_or_not, metrics, split_data
from preprocess_data import df_group_by_session_id

import os
import sys


if __name__ == '__main__':
    assert len(sys.argv) == 7, ('Incorrect argument list, names also shouldn\'t contain spaces' + '\n' + 'usage:'
                                'argv[0] /dir/with/data clicks_file buys_file test_file scores_file resulting_file')

    path = sys.argv[1]
    file_clicks_basename = sys.argv[2]
    file_buys_basename = sys.argv[3]
    file_test_basename = sys.argv[4]
    file_scores_basename = sys.argv[5]
    file_result_basename = sys.argv[6]

    file_clicks = os.path.join(path, file_clicks_basename)
    file_buys = os.path.join(path, file_buys_basename)
    file_test = os.path.join(path, file_test_basename)
    file_scores = os.path.join(path, file_scores_basename)
    file_result = os.path.join(path, file_result_basename)

    effective_columns_names = ['Session ID', 'Timestamp', 'Item ID']
    clicks = read_clicks(file_clicks, usecols=effective_columns_names)
    buys = read_buys(file_buys, usecols=effective_columns_names)

    clicks.sort_values('Timestamp', inplace=True)
    clicks_grouped_by_session_id, clicks_grouped_by_session_id_keys = df_group_by_session_id(clicks)
    _, buys_grouped_by_session_id_keys = df_group_by_session_id(buys)

    what_to_buy = extract_what_to_buy(clicks)
    buy_or_not = extract_buy_or_not(clicks, what_to_buy)
    buys_result = extract_buys(clicks_grouped_by_session_id_keys, buys_grouped_by_session_id_keys)

    buy_or_not_train, buy_or_not_val, buys_result_train, buys_result_val = split_data(buy_or_not, buys_result, 0.2)
    classifier = fit_data(buy_or_not_train, buys_result_train)

    predictions_val = predict_buy_or_not(classifier, buy_or_not_val)
    scores = metrics(buys_result_val, predictions_val)
    write_metrics(scores, file_scores)

    test = read_clicks(file_test, usecols=effective_columns_names)
    test_grouped_by_session_id, _ = df_group_by_session_id(test)

    what_to_buy_test = extract_what_to_buy(test_grouped_by_session_id)
    buy_or_not_test = extract_buy_or_not(test_grouped_by_session_id, what_to_buy_test)

    predictions_test = predict_buy_or_not(classifier, buy_or_not_test)
    write_predictions(predictions_test, file_result)
