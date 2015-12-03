__author__ = 'pdanilov'

from read_and_write_data import read_clicks, read_buys, write_predictions, features_to_csv
from feature_extraction import extract_buy_or_not, extract_what_to_buy, extract_buys
from predictions import fit_data, predict_buy_or_not

import os
import sys


if __name__ == '__main__':
    if len(sys.argv) != 6:
        args_err_msg = r"Incorrect argument list, names also shouldn't contain spaces" + \
                       "\nusage: argv[0] /dir/with/data clicks_file buys_file test_file resulting_file"
        print(args_err_msg)
        exit(0)
    path = sys.argv[1]
    file_clicks_basename = sys.argv[2]
    file_buys_basename = sys.argv[3]
    file_test_basename = sys.argv[4]
    file_result_basename = sys.argv[5]

    file_clicks = os.path.join(path, file_clicks_basename)
    file_buys = os.path.join(path, file_buys_basename)
    file_test = os.path.join(path, file_test_basename)
    file_result = os.path.join(path, file_result_basename)

    print('Reading clicks file')
    clicks = read_clicks(file_clicks)
    print('Reading buys file')
    buys = read_buys(file_buys)

    print('Extracting what-to-buy train')
    what_to_buy = extract_what_to_buy(clicks)
    print('Extracting buy-or-not train')
    buy_or_not = extract_buy_or_not(clicks, what_to_buy)
    print('Extracting buys train')
    buys_result = extract_buys(clicks, buys)

    # features_to_csv(what_to_buy, buy_or_not, path)

    print('Classifier learning')
    classifier = fit_data(buy_or_not, buys_result)

    print('Reading test file')
    test = read_clicks(file_test)

    print('Extracting what-to-buy test')
    what_to_buy_test = extract_what_to_buy(test)
    print('Extracting buy-or-not test')
    buy_or_not_test = extract_buy_or_not(test, what_to_buy_test)

    print('Prediction')
    predictions = predict_buy_or_not(classifier, buy_or_not_test)

    print('Output predictions')
    write_predictions(predictions, file_result)

