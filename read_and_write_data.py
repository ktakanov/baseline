__author__ = 'pdanilov'

import pandas as pd
import numpy as np
import os

date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
effective_columns_name_list = ['Session ID', 'Timestamp', 'Item ID']


def date_parse(dates):
    return [pd.datetime.strptime(date, date_format) for date in dates]


def read_clicks(file_clicks):
    column_names = ['Session ID', 'Timestamp', 'Item ID', 'Category']
    dict_type = {'Session ID': np.int32, 'Timestamp': pd.tslib.Timestamp, 'Item ID': np.int32, 'Category': str}
    global date_format
    return pd.read_csv(file_clicks, engine='c', names=column_names, dtype=dict_type, parse_dates=['Timestamp'], date_parser=date_parse, \
                       usecols=effective_columns_name_list)


def read_buys(file_buys):
    column_names = ['Session ID', 'Timestamp', 'Item ID', 'Price', 'Quantity']
    dict_type = {'Session ID': np.int32, 'Timestamp': pd.tslib.Timestamp, 'Item ID': np.int32, 'Price': np.int16, 'Quantity': np.int8}
    global date_format
    return pd.read_csv(file_buys, engine='c', names=column_names, dtype=dict_type, parse_dates=['Timestamp'], date_parser=date_parse, \
                       usecols=effective_columns_name_list)

def write_predictions(predictions, file_predictions):
    np.savetxt(file_predictions, predictions, fmt='%d', newline='\n')


def write_metrics(metrics, file_name):
    file = open(file_name, 'w')
    file.write('Precision: {0}\nRecall: {1}\nAccuracy: {2}'.format(metrics[0], metrics[1], metrics[2]))
    file.close()


def features_to_csv(what_to_buy_features, buy_or_not_features, path_to_data):
    for key in what_to_buy_features.keys():
        what_to_buy_features[key].to_csv(path_or_buf=os.path.join(path_to_data, 'features', key + '.dat'))

    buy_or_not_features.to_csv(path_or_buf=os.path.join(path_to_data, 'features', 'p', '.dat'))
