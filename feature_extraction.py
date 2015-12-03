__author__ = 'pdanilov'

import pandas as pd
import numpy as np

effective_columns_name_list = ['Session ID', 'Timestamp', 'Item ID']


def extract_p1(grouped):
    return grouped['Timestamp'].count()


def extract_p2(f3):
    return f3.mean(axis='columns')


def extract_p3(grouped):
    return grouped.apply(lambda x: x['Timestamp'].iloc[-1] - x['Timestamp'].iloc[0]).astype('timedelta64[ms]')


def extract_p4(grouped):
    group_keys = grouped.groups.keys()
    dict_of_series = {key: grouped.get_group(key)['Timestamp'].diff().astype('timedelta64[ms]').max() for key in group_keys}
    return pd.Series(dict_of_series)


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


def extract_buy_or_not(clicks, features_what_to_buy):
    clicks = clicks[effective_columns_name_list]

    clicks_time_sorted = clicks.sort('Timestamp')
    grouped_clicks_time_sorted = clicks_time_sorted.groupby('Session ID')

    p1 = extract_p1(grouped_clicks_time_sorted)
    p2 = extract_p2(features_what_to_buy['f3'])
    p3 = extract_p3(grouped_clicks_time_sorted)
    p4 = extract_p4(grouped_clicks_time_sorted)
    p5 = extract_p5(grouped_clicks_time_sorted, p3)
    p6 = extract_p6(features_what_to_buy['f3'])
    p10 = extract_p6(features_what_to_buy['f6'])
    p11 = extract_p6(features_what_to_buy['f7'])

    return np.matrix([p1, p2, p3, p4, p5, p6, p10, p11], dtype=np.float64).transpose()


# computation of feature matrix for data frames being constructed (features F1...F7)
def get_matrix_by_dicts(n_rows, n_cols, min_id, list_of_dicts, fill_nans=False):
    # would be fine trying to find out how to optimize this function using np.concatenate
    # data_type = type(list_of_dicts[0][list(list_of_dicts[0].keys())[0]])
    if fill_nans:
        matrix = np.empty((n_rows, n_cols))
        matrix.fill(np.nan)
    else:
        matrix = np.zeros((n_rows, n_cols))
    for row, column_dict in enumerate(list_of_dicts):
        for col, value in column_dict.items():
            matrix[row][col - min_id] = value  # shift second axis
    return matrix


# getting result of computations for every feature in F1...F7 into data frames
def get_resulting_data_frame(grouped, func, fill_nans=False):
    group_keys = grouped.groups.keys()
    min_item_id = grouped['Item ID'].min().min()
    max_item_id = grouped['Item ID'].max().max()
    num_of_items = max_item_id - min_item_id + 1
    num_of_sessions = len(group_keys)
    list_of_dicts = [func(grouped.get_group(key)) for key in group_keys]
    matrix = get_matrix_by_dicts(num_of_sessions, num_of_items, min_item_id, list_of_dicts, fill_nans)
    return pd.DataFrame(matrix, index=list(group_keys), columns=list(range(min_item_id, max_item_id + 1)))


def extract_f1(grouped):
    return get_resulting_data_frame(grouped, lambda x: {x['Item ID'].iloc[0]: True})


def extract_f2(grouped):
    return get_resulting_data_frame(grouped, lambda x: {x['Item ID'].iloc[-1]: True})


def extract_f3(grouped):
    return get_resulting_data_frame(grouped, lambda x: x['Item ID'].value_counts().to_dict(), fill_nans=True)


def extract_f4(grouped):
    def time_spent(group):
        # differing starts from first, not second row, with opposite direction -> multiply by -1
        time_diff = group['Timestamp'].diff(periods=-1).astype('timedelta64[ms]') * (-1)
        # time_diff.iloc[-1] = time_diff.mean() would be equivalent to next line
        time_diff.fillna(time_diff.mean(), inplace=True)
        group.insert(loc=group.shape[1], column='Time Difference', value=time_diff)
        return group.groupby('Item ID')['Time Difference'].sum().to_dict()

    return get_resulting_data_frame(grouped, time_spent)


def extract_f5(grouped):
    pass


def extract_f6(grouped):
    def sequent_clicks(group):
        diff = group['Item ID'].diff()
        diff_reverse = group['Item ID'].diff(periods=-1)
        group.insert(loc=group.shape[1], column='Diff', value=diff)
        group.insert(loc=group.shape[1], column='Diff Reverse', value=diff_reverse)
        repeated = group.apply(lambda x: 1 if (x['Diff'] == 0 or x['Diff Reverse'] == 0) else 0, axis='columns')
        group.insert(loc=group.shape[1], column='Repeated Click', value=repeated)
        return group.groupby('Item ID')['Repeated Click'].sum().apply(lambda x: 1 if x == 0 else x).to_dict()

    return get_resulting_data_frame(grouped, sequent_clicks)


def extract_f7(grouped):
    def max_duration_between_sequent_clicks(group):
        positive_key = 1
        negative_key = 0
        repeated = group['Item ID'].diff().apply(lambda x: positive_key if x == 0 else negative_key)
        time_diff = group['Timestamp'].diff().astype('timedelta64[ms]')
        group.insert(loc=group.shape[1], column='Time Difference', value=time_diff)
        group.insert(loc=group.shape[1], column='Repeated Click', value=repeated)
        group_by_repeated = group.groupby('Repeated Click')
        result = group_by_repeated.get_group(positive_key).groupby('Item ID')['Time Difference'].max().to_dict() \
            if positive_key in group_by_repeated.groups.keys() else {group['Item ID'].iloc[0]: np.float64(0)}
        return result

    return get_resulting_data_frame(grouped, max_duration_between_sequent_clicks)


def extract_what_to_buy(clicks):
    clicks = clicks[effective_columns_name_list]

    clicks_time_sorted = clicks.sort('Timestamp')
    grouped_clicks_time_sorted = clicks_time_sorted.groupby('Session ID')

    # f1 = extract_f1(grouped_clicks_time_sorted)
    # f2 = extract_f2(grouped_clicks_time_sorted)
    f3 = extract_f3(grouped_clicks_time_sorted)
    # f4 = extract_f4(grouped_clicks_time_sorted)
    f6 = extract_f6(grouped_clicks_time_sorted)
    f7 = extract_f7(grouped_clicks_time_sorted)

    return {'f3': f3, 'f6': f6, 'f7': f7}


def extract_buys(clicks, buys):
    clicks_grouped = clicks.groupby('Session ID')
    clicks_session_id_list = list(clicks_grouped.groups.keys())
    session_id_min = min(clicks_session_id_list)
    session_id_max = max(clicks_session_id_list)

    buys_grouped = buys.groupby('Session ID')
    buys_session_id_dict = [{key: 1} for key in buys_grouped.groups.keys()]

    array = get_matrix_by_dicts(session_id_max - session_id_min + 1, 1, session_id_min, buys_session_id_dict)

    columns = ['Session ID', 'Prediction']
    resulting_df = pd.DataFrame([range(session_id_min, session_id_max + 1), array], columns=columns)

    return resulting_df
