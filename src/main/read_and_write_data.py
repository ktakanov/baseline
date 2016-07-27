import pandas as pd
import numpy as np
import os


def read_data(file_name, column_names, dtype_dict, usecols=None):
    return pd.read_csv(file_name, engine='c', names=column_names, dtype=dtype_dict, parse_dates=['Timestamp'],
                       usecols=usecols, date_parser=lambda date: pd.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%fZ'))


def read_clicks(file_clicks, usecols=None):
    column_names = ['Session ID', 'Timestamp', 'Item ID', 'Category']
    dtype_dict = {'Session ID': np.int32,
                  'Timestamp': pd.tslib.Timestamp,
                  'Item ID': np.int32,
                  'Category': str}

    return read_data(file_clicks, column_names, dtype_dict, usecols)


def read_buys(file_buys, usecols=None):
    column_names = ['Session ID', 'Timestamp', 'Item ID', 'Price', 'Quantity']
    dtype_dict = {'Session ID': np.int32,
                  'Timestamp': pd.tslib.Timestamp,
                  'Item ID': np.int32,
                  'Price': np.int16,
                  'Quantity': np.int8}

    return read_data(file_buys, column_names, dtype_dict, usecols)


def write_predictions(predictions, file_predictions):
    np.savetxt(file_predictions, predictions, fmt='%d', newline='\n')


def write_metrics(metrics, file_name):
    with open(file_name, 'w') as out_file:
        out_file.write('Precision: {0}\nRecall: {1}\nF1-Score: {2}\nAccuracy: {3}'.format(metrics[0], metrics[1],
                                                                                      metrics[2], metrics[3]))


def features_to_csv(what_to_buy_features, buy_or_not_features, path_to_data):
    for key in what_to_buy_features.keys():
        what_to_buy_features[key].to_csv(path_or_buf=os.path.join(path_to_data, 'features', key + '.dat'))

    buy_or_not_features.to_csv(path_or_buf=os.path.join(path_to_data, 'features', 'p', '.dat'))
