import pandas as pd
import numpy as np


def extract_what_to_buy(clicks_gb):
    f3 = clicks_gb.apply(counts)
    f6 = clicks_gb.apply(sequent_clicks)
    f7 = clicks_gb.apply(max_duration_between_sequent_clicks)
    f3, f6, f7 = map(lambda df: df.reset_index().rename(columns={'level_1': 'Item ID'}), [f3, f6, f7])

    return {'f3': f3, 'f6': f6, 'f7': f7}


def get_df_features(grouped, func):
    features_list = []

    for session_id in grouped.groups.iterkeys():
        session_grouped_by_items = grouped.get_group(session_id)
        group_func = func(session_grouped_by_items).reset_index().rename(columns={'index': 'Item ID'})
        group_func.insert(loc=0, column='Session ID', value=session_id)
        features_list.append(group_func)

    return pd.concat(features_list, ignore_index=True)


def counts(group):
    return pd.Series(group['Item ID'].value_counts(), name='Counts')


def sequent_clicks(group):
    diff = group['Item ID'].diff()
    diff_reverse = group['Item ID'].diff(periods=-1)
    repeated = ((diff == 0) | (diff_reverse == 0)).apply(lambda x: np.int32(x))
    group.insert(loc=group.shape[1], column='Sequent Clicks', value=repeated)

    return group\
        .groupby('Item ID')['Sequent Clicks']\
        .sum()\
        .apply(lambda x: 1 if x == 0 else x)


def max_duration_between_sequent_clicks(group):
    positive_key = 1
    negative_key = 0
    repeated = group['Item ID'].diff().apply(lambda x: positive_key if x == 0 else negative_key)
    time_diff = group['Timestamp'].diff().astype('timedelta64[ms]')
    group.insert(loc=group.shape[1], column='Time Difference', value=time_diff)
    group.insert(loc=group.shape[1], column='Repeated Click', value=repeated)
    group_by_repeated = group.groupby('Repeated Click')

    result_pos =\
        group_by_repeated.get_group(positive_key).groupby('Item ID')['Time Difference'].max()\
        if positive_key in group_by_repeated.groups.keys()\
        else\
        None

    result_neg =\
        group_by_repeated.get_group(negative_key).groupby('Item ID')['Time Difference'].apply(lambda x: np.float32(0))\
        if negative_key in group_by_repeated.groups.keys()\
        else\
        None

    return pd.concat([result_pos, result_neg])


def extract_buy_or_not(clicks_gb, features_what_to_buy):
    p1 = clicks_gb.size()
    p2 = features_what_to_buy['f3'].groupby('Session ID')['Counts'].mean()
    p3 = clicks_gb.apply(lambda x: x['Timestamp'].iloc[-1] - x['Timestamp'].iloc[0]).astype('timedelta64[ms]')
    p4 = clicks_gb.apply(lambda x: x['Timestamp'].diff().astype('timedelta64[ms]').fillna(np.float64(0)).max())
    p5 = p3.divide(p1)
    p6 = features_what_to_buy['f3'].groupby('Session ID')['Counts'].max()
    p10 = features_what_to_buy['f6'].groupby('Session ID')['Sequent Clicks'].max()
    p11 = features_what_to_buy['f7'].groupby('Session ID')['Time Difference'].max()

    return np.array([p1, p2, p3, p4, p5, p6, p10, p11], dtype=np.float32).transpose()


def extract_buys(clicks_group_keys, buys_group_keys):
    session_series = pd.Series(0, dtype=np.int8, index=clicks_group_keys)
    session_series[buys_group_keys] = np.int8(1)

    return np.array(session_series)
