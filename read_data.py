__author__ = 'pdanilov'

import pandas as pd

date_format = '%Y-%m-%dT%H:%M:%S.%fZ'


def date_parse(dates):
    return [pd.datetime.strptime(date, date_format) for date in dates]


def read_clicks(file_clicks):
    column_names = ['Session ID', 'Timestamp', 'Item ID', 'Category']
    dict_type = {'Session ID': int, 'Timestamp': pd.tslib.Timestamp, 'Item ID': int, 'Category': str}
    global date_format
    return pd.read_csv(file_clicks, engine='c', names=column_names, dtype=dict_type, parse_dates=['Timestamp'], date_parser=date_parse)


def read_buys(file_buys):
    column_names = ['Session ID', 'Timestamp', 'Item ID', 'Price', 'Quantity']
    dict_type = {'Session ID': int, 'Timestamp': pd.tslib.Timestamp, 'Item ID': int, 'Price': int, 'Quantity': int}
    global date_format
    return pd.read_csv(file_buys, engine='c', names=column_names, dtype=dict_type, parse_dates=['Timestamp'], date_parser=date_parse)
