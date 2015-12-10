__author__ = 'pdanilov'

import pandas as pd
import numpy as np


def extract_p1(grouped):
    return grouped.size()


def extract_p2(f3):
    return f3.groupby('Session ID')['Counts'].mean()


def extract_p3(grouped):
    return grouped.apply(lambda x: x['Timestamp'].iloc[-1] - x['Timestamp'].iloc[0]).astype('timedelta64[ms]')


def extract_p4(grouped):
    return grouped.apply(lambda x: x['Timestamp'].diff().astype('timedelta64[ms]').fillna(np.float64(0)).max())


def extract_p5(p1, p3):
    return p3.divide(p1)


def extract_p6(f3):
    return f3.groupby('Session ID')['Counts'].max()


def extract_p10(f6):
    return f6.groupby('Session ID')['Repeated Click'].max()


def extract_p11(f7):
    return f7.groupby('Session ID')['Time Difference'].max()


def extract_buy_or_not(clicks_gb, features_what_to_buy):
    print('\tExtracting p1')
    p1 = extract_p1(clicks_gb)
    print('\tExtracting p2')
    p2 = extract_p2(features_what_to_buy['f3'])
    print('\tExtracting p3')
    p3 = extract_p3(clicks_gb)
    print('\tExtracting p4')
    p4 = extract_p4(clicks_gb)
    print('\tExtracting p5')
    p5 = extract_p5(p1, p3)
    print('\tExtracting p6')
    p6 = extract_p6(features_what_to_buy['f3'])
    print('\tExtracting p10')
    p10 = extract_p10(features_what_to_buy['f6'])
    print('\tExtracting p11')
    p11 = extract_p11(features_what_to_buy['f7'])

    features_matrix = np.matrix([p1, p2, p3, p4, p5, p6, p10, p11]).transpose()
    return features_matrix


def create_session_data_frame(df, func, session_id):
    result = func(df).reset_index()
    result.insert(loc=result.shape[1], column='Session ID', value=session_id)
    return result


def get_resulting_data_frame(grouped, group_keys, func):
    num_of_sessions = len(group_keys)
    list_of_session_data_frames = [None] * num_of_sessions
    for index in range(num_of_sessions):
        key = group_keys[index]
        list_of_session_data_frames[index] = create_session_data_frame(grouped.get_group(key), func, np.int32(key))
    return pd.concat(list_of_session_data_frames, ignore_index=True)


def extract_f3(grouped, group_keys):
    def value_counts(group):
        return pd.Series(group['Item ID'].value_counts(), name='Counts')

    return get_resulting_data_frame(grouped, group_keys, value_counts).rename(columns={'index': 'Item ID'}).drop('Item ID', axis=1)


def extract_f6(grouped, group_keys):
    def sequent_clicks(group):
        diff = group['Item ID'].diff()
        diff_reverse = group['Item ID'].diff(periods=-1)
        group.insert(loc=group.shape[1], column='Diff', value=diff)
        group.insert(loc=group.shape[1], column='Diff Reverse', value=diff_reverse)
        repeated = group.apply(lambda x: 1 if (x['Diff'] == 0 or x['Diff Reverse'] == 0) else 0, axis='columns')
        # Feature is Repeated Click
        group.insert(loc=group.shape[1], column='Repeated Click', value=repeated)
        result = group.groupby('Item ID')['Repeated Click'].sum().apply(lambda x: 1 if x == 0 else x)
        return result

    return get_resulting_data_frame(grouped, group_keys, sequent_clicks).drop('Item ID', axis=1)


def extract_f7(grouped, group_keys):
    def max_duration_between_sequent_clicks(group):
        positive_key = 1
        negative_key = 0
        repeated = group['Item ID'].diff().apply(lambda x: positive_key if x == 0 else negative_key)
        time_diff = group['Timestamp'].diff().astype('timedelta64[ms]')
        # Feature is Time Difference
        group.insert(loc=group.shape[1], column='Time Difference', value=time_diff)
        group.insert(loc=group.shape[1], column='Repeated Click', value=repeated)
        group_by_repeated = group.groupby('Repeated Click')
        result_pos = group_by_repeated.get_group(positive_key).groupby('Item ID')['Time Difference'].max() \
            if positive_key in group_by_repeated.groups.keys() else None
        result_neg = group_by_repeated.get_group(negative_key).groupby('Item ID')['Time Difference'].apply(lambda x: np.float32(0)) \
            if negative_key in group_by_repeated.groups.keys() else None
        return pd.concat([result_pos, result_neg])

    return get_resulting_data_frame(grouped, group_keys, max_duration_between_sequent_clicks).drop('Item ID', axis=1)


def extract_what_to_buy(clicks_gb, clicks_group_keys):
    print('\tExtracting f3')
    f3 = extract_f3(clicks_gb, clicks_group_keys)
    print('\tExtracting f6')
    f6 = extract_f6(clicks_gb, clicks_group_keys)
    print('\tExtracting f7')
    f7 = extract_f7(clicks_gb, clicks_group_keys)

    return {'f3': f3, 'f6': f6, 'f7': f7}


def extract_buys(clicks_group_keys, buys_group_keys):
    session_series = pd.Series(0, dtype=np.int8, index=clicks_group_keys)
    session_series[buys_group_keys] = np.int8(1)

    result = np.matrix(session_series).transpose()

    return result
