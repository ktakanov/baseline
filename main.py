__author__ = 'pdanilov'

from read_and_write_data import read_clicks, read_buys, write_predictions, write_metrics
from feature_extraction import extract_buy_or_not, extract_what_to_buy, extract_buys
from predictions import fit_data, predict_buy_or_not, metrics, split_data
from preprocess_data import slice_data, sort_and_group_by

import os
import sys


if __name__ == '__main__':
    if len(sys.argv) != 7:
        args_err_msg = r"Incorrect argument list, names also shouldn't contain spaces" + \
                       "\nusage: argv[0] /dir/with/data clicks_file buys_file test_file scores_file resulting_file"
        print(args_err_msg)
        exit(0)
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

    print('Reading clicks file')
    clicks = read_clicks(file_clicks)
    print('Reading buys file')
    buys = read_buys(file_buys)

    print('Slicing data')
    slice_data(clicks, buys, frac=0.01)

    print('Sort and groupby')
    # after applying clicks, buys are groupby objects
    clicks, buys = sort_and_group_by(clicks, buys)

    print('Extracting what-to-buy train')
    what_to_buy = extract_what_to_buy(clicks)
    print('Extracting buy-or-not train')
    buy_or_not = extract_buy_or_not(clicks, what_to_buy)
    print('Extracting buys train')
    buys_result = extract_buys(clicks, buys)

    print('Classifier learning on validation')
    [buy_or_not_train, buy_or_not_val, buys_result_train, buys_result_val] = split_data(buy_or_not, buys_result)
    classifier = fit_data(buy_or_not_train, buys_result_train)

    print('Prediction on validation')
    predictions_val = predict_buy_or_not(classifier, buy_or_not_val)
    scores = metrics(buys_result_val, predictions_val)
    write_metrics(scores, file_scores)

    print('Reading test file')
    test = read_clicks(file_test)

    print('Extracting what-to-buy test')
    what_to_buy_test = extract_what_to_buy(test)
    print('Extracting buy-or-not test')
    buy_or_not_test = extract_buy_or_not(test, what_to_buy_test)

    print('Classifier learning')
    classifier = fit_data(buy_or_not, buys_result)

    print('Prediction')
    predictions_test = predict_buy_or_not(classifier, buy_or_not_test)

    print('Output predictions')
    write_predictions(predictions_test, file_result)

