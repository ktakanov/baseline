__author__ = 'pdanilov'

import pandas as pd
import numpy as np


def extract_p1(grouped):
    return grouped['Timestamp'].count()


def extract_p2(f3):
    return f3.mean(axis='columns')


def extract_p3(grouped):
    return grouped.apply(lambda x: x['Timestamp'].iloc[-1] - x['Timestamp'].iloc[0]).astype('timedelta64[ms]')


def extract_p4(grouped, group_keys):
    list_of_series = [grouped.get_group(key)['Timestamp'].diff().astype('timedelta64[ms]').max() for key in group_keys]
    return pd.Series(list_of_series, index=group_keys, dtype=np.float16)


def extract_p5(grouped, p3):
    return p3.divide(grouped.size())


def extract_p6(f3):
    return f3.max(axis='columns')


def extract_p7(f5):
    return f5.mean(axis='columns')


def extract_p8(f5):
    return f5.max(axis='columns')


def extract_p9(grouped):
    # no need to implement computation of this feature
    pass


def extract_p10(f6):
    return f6.max(axis='columns')


def extract_p11(f7):
    return f7.max(axis='columns')


def extract_buy_or_not(clicks_gb, clicks_group_keys, features_what_to_buy):
    print('\tExtracting p1')
    p1 = extract_p1(clicks_gb)
    print('\tExtracting p2')
    p2 = extract_p2(features_what_to_buy['f3'])
    print('\tExtracting p3')
    p3 = extract_p3(clicks_gb)
    print('\tExtracting p4')
    p4 = extract_p4(clicks_gb, clicks_group_keys)
    print('\tExtracting p5')
    p5 = extract_p5(clicks_gb, p3)
    print('\tExtracting p6')
    p6 = extract_p6(features_what_to_buy['f3'])
    print('\tExtracting p10')
    p10 = extract_p6(features_what_to_buy['f6'])
    print('\tExtracting p11')
    p11 = extract_p6(features_what_to_buy['f7'])

    return np.matrix([p1, p2, p3, p4, p5, p6, p10, p11], dtype=np.float16).transpose()


# computation of feature matrix for data frames being constructed (features F1...F7)
def get_matrix_by_dicts(n_rows, n_cols, min_id, list_of_dicts, fill_with_nans=False):
    # would be fine trying to find out how to optimize this function using np.concatenate
    # data_type = type(list_of_dicts[0][list(list_of_dicts[0].keys())[0]])
    if fill_with_nans:
        matrix = np.empty((n_rows, n_cols))
        matrix.fill(np.nan)
    else:
        matrix = np.zeros((n_rows, n_cols))
    for row, column_dict in enumerate(list_of_dicts):
        for col, value in column_dict.items():
            matrix[row][col - min_id] = value  # shift second axis
    return matrix


def create_row(matrix, df, func, min_item_id, dtype, fill_with_nans=False):
    result = func(df)
    index = [] if len(result.index) == 0 else (result.index - min_item_id)
    num_of_items = matrix.shape[1]
    row = np.zeros(num_of_items, dtype=dtype)
    if fill_with_nans:
        row.fill(np.nan)
    row[index] = result.as_matrix()
    return row

# getting result of computations for every feature in F1...F7 into data frames
def get_resulting_data_frame(grouped, group_keys, func, dtype, fill_with_nans=False):
    min_item_id = grouped['Item ID'].min().min()
    max_item_id = grouped['Item ID'].max().max()
    num_of_items = max_item_id - min_item_id + 1
    num_of_sessions = len(group_keys)
    matrix = np.zeros([num_of_sessions, num_of_items], dtype=dtype)
    for index in range(num_of_sessions):
        key = group_keys[index]
        matrix[index] = create_row(matrix, grouped.get_group(key), func, min_item_id, dtype, fill_with_nans)
    return pd.DataFrame(matrix, index=list(group_keys), columns=list(range(min_item_id, max_item_id + 1)))


def extract_f1(grouped, group_keys):
    return get_resulting_data_frame(grouped, group_keys, lambda x: pd.DataFrame([True], index=[x['Item ID'].iloc[0]], dtype=bool), dtype=bool)


def extract_f2(grouped, group_keys):
    return get_resulting_data_frame(grouped, group_keys, lambda x: pd.DataFrame([True], index=[x['Item ID'].iloc[-1]], dtype=bool), dtype=bool)


def extract_f3(grouped, group_keys):
    return get_resulting_data_frame(grouped, group_keys, lambda x: x['Item ID'].value_counts(), dtype=np.float16, fill_with_nans=True)


def extract_f4(grouped, group_keys):
    def time_spent(group):
        # differing starts from first, not second row, with opposite direction -> multiply by -1
        time_diff = group['Timestamp'].diff(periods=-1).astype('timedelta64[ms]') * (-1)
        # time_diff.iloc[-1] = time_diff.mean() would be equivalent to next line
        time_diff.fillna(time_diff.mean(), inplace=True)
        group.insert(loc=group.shape[1], column='Time Difference', value=time_diff)
        return group.groupby('Item ID')['Time Difference'].sum()

    return get_resulting_data_frame(grouped, group_keys, time_spent, dtype=np.float16)


def extract_f5(grouped, group_keys):
    pass


def extract_f6(grouped, group_keys):
    def sequent_clicks(group):
        diff = group['Item ID'].diff()
        diff_reverse = group['Item ID'].diff(periods=-1)
        group.insert(loc=group.shape[1], column='Diff', value=diff)
        group.insert(loc=group.shape[1], column='Diff Reverse', value=diff_reverse)
        repeated = group.apply(lambda x: 1 if (x['Diff'] == 0 or x['Diff Reverse'] == 0) else 0, axis='columns')
        group.insert(loc=group.shape[1], column='Repeated Click', value=repeated)
        return group.groupby('Item ID')['Repeated Click'].sum().apply(lambda x: 1 if x == 0 else x)

    return get_resulting_data_frame(grouped, group_keys, sequent_clicks, dtype=np.int16)


def extract_f7(grouped, group_keys):
    def max_duration_between_sequent_clicks(group):
        positive_key = 1
        negative_key = 0
        repeated = group['Item ID'].diff().apply(lambda x: positive_key if x == 0 else negative_key)
        time_diff = group['Timestamp'].diff().astype('timedelta64[ms]')
        group.insert(loc=group.shape[1], column='Time Difference', value=time_diff)
        group.insert(loc=group.shape[1], column='Repeated Click', value=repeated)
        group_by_repeated = group.groupby('Repeated Click')
        result = group_by_repeated.get_group(positive_key).groupby('Item ID')['Time Difference'].max() \
            if positive_key in group_by_repeated.groups.keys() else pd.DataFrame(dtype=np.float16)
        return result

    return get_resulting_data_frame(grouped, group_keys, max_duration_between_sequent_clicks, dtype=np.float16)


def extract_what_to_buy(clicks_gb, clicks_group_keys):
    print('\tExtracting f3')
    f3 = extract_f3(clicks_gb, clicks_group_keys)
    print('\tExtracting f6')
    f6 = extract_f6(clicks_gb, clicks_group_keys)
    print('\tExtracting f7')
    f7 = extract_f7(clicks_gb, clicks_group_keys)

    return {'f3': f3, 'f6': f6, 'f7': f7}


def extract_buys(clicks_group_keys, buys_group_keys):
    clicks_session_id_list = list(clicks_group_keys)
    min_session_id = min(clicks_session_id_list)
    max_session_id = max(clicks_session_id_list)
    num_of_sessions = max_session_id - min_session_id + 1

    array = np.zeros(num_of_sessions, dtype=np.int8)
    index = np.array(list(buys_group_keys), dtype=np.int32) - min_session_id
    array[index] = np.int8(1)

    columns = ['Session ID', 'Prediction']
    resulting_df = pd.DataFrame([range(min_session_id, max_session_id + 1), array], columns=columns)

    return resulting_df
