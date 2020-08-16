import pandas as pd
import numpy as np
import os

date_format = '%Y-%m-%dT%H:%M:%S.%fZ'


def read_data(file_name, column_names, dtype_dict, usecols=None):
    return pd.read_csv(file_name, engine='c', names=column_names, dtype=dtype_dict, parse_dates=['Timestamp'],
                       usecols=usecols, date_parser=lambda date: pd.datetime.strptime(date, date_format))


def read_clicks(file_clicks, usecols=None):
    column_names = ['Session ID', 'Timestamp', 'Item ID', 'Category']
    dtype_dict = {'Session ID': np.int32,
                  'Timestamp': pd.Timestamp,
                  'Item ID': np.int32,
                  'Category': str}

    return read_data(file_clicks, column_names, dtype_dict, usecols)


def read_buys(file_buys, usecols=None):
    column_names = ['Session ID', 'Timestamp', 'Item ID', 'Price', 'Quantity']
    dtype_dict = {'Session ID': np.int32,
                  'Timestamp': pd.Timestamp,
                  'Item ID': np.int32,
                  'Price': np.int32,
                  'Quantity': np.int8}

    return read_data(file_buys, column_names, dtype_dict, usecols)


def mkdir_opt(full_file_path):
    file_dir, _ = os.path.split(full_file_path)

    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)


def write_predictions(file_predictions, predictions):
    mkdir_opt(file_predictions)
    np.savetxt(file_predictions, predictions, fmt='%d', newline='\n')


def write_metrics(file_name, metrics):
    mkdir_opt(file_name)

    with open(file_name, 'w') as out_file:
        out_file.write('Precision: {0}\nRecall: {1}\nF1-Score: {2}\nAccuracy: {3}'.format(metrics[0], metrics[1],
                                                                                          metrics[2], metrics[3]))


def write_df(out_file, df):
    mkdir_opt(out_file)
    df.to_csv(out_file, na_rep='NA', header=False, index=False, date_format=date_format)


def features_to_csv(file_name, features_df_list):
    features_df_list = [features_df.sort_values(by=['Session ID', 'Item ID']) for features_df in features_df_list]
    features_df_concat = features_df_list[0].iloc[:, :2]

    for features_df in features_df_list:
        features_df_concat = features_df_concat.join(features_df.iloc[:, 2])

    mkdir_opt(file_name)
    features_df_concat.to_csv(path_or_buf=file_name, index=False)


def features_from_csv(file_name, common_col_names, features_col_names_dict):
    features_df = pd.read_csv(file_name, engine='c')
    features_df_dict = features_col_names_dict

    for feature_name, col_name in features_col_names_dict.items():
        features_df_dict[feature_name] = features_df[common_col_names + [col_name]]

    return features_df_dict
