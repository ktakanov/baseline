import pandas as pd
import numpy as np


def extract_what_to_buy(clicks_gb):
    f3 = extract_f3(clicks_gb)
    f6 = extract_f6(clicks_gb)
    f7 = extract_f7(clicks_gb)

    return {'f3': f3, 'f6': f6, 'f7': f7}


def extract_f3(clicks_gb):

    def counts(group):
        return pd.Series(group['Item ID'].value_counts(), name='Counts')

    return clicks_gb.apply(counts).reset_index().rename(columns={'level_1': 'Item ID'})


def extract_f6(clicks_gb):

    def sequent_clicks(group):
        diff = group['Item ID'].diff()
        diff_reverse = group['Item ID'].diff(periods=-1)
        repeated = ((diff == 0) | (diff_reverse == 0)).apply(lambda x: np.int32(x))
        group.insert(loc=group.shape[1], column='Sequent Clicks', value=repeated)

        return group\
            .groupby('Item ID')['Sequent Clicks']\
            .sum()\
            .apply(lambda x: 1 if x == 0 else x)

    return clicks_gb.apply(sequent_clicks).reset_index().rename(columns={'level_1': 'Item ID'})


def extract_f7(clicks_gb):

    def max_duration_between_sequent_clicks(group):
        positive_key = np.int8(1)
        negative_key = np.int8(0)
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
            group_by_repeated.get_group(negative_key).groupby('Item ID')['Time Difference']\
                .apply(lambda x: np.float64(np.iinfo(np.int64).max))\
            if negative_key in group_by_repeated.groups.keys()\
            else\
            None

        result = pd.concat([result_pos, result_neg]).sort_values()

        return result[~result.index.duplicated()]

    return clicks_gb.apply(max_duration_between_sequent_clicks).reset_index().rename(columns={'level_1': 'Item ID'})


def extract_buy_or_not(clicks_gb, features_what_to_buy):
    p1 = extract_p1(clicks_gb)
    p2 = extract_p2(features_what_to_buy['f3'])
    p3 = extract_p3(clicks_gb)
    p4 = extract_p4(clicks_gb)
    p5 = extract_p5(p1, p3)
    p6 = extract_p2(features_what_to_buy['f3'])
    p10 = extract_p10(features_what_to_buy['f6'])
    p11 = extract_p11(features_what_to_buy['f7'])

    return np.array([p1, p2, p3, p4, p5, p6, p10, p11], dtype=np.float32).transpose()


def extract_p1(clicks_gb):
    return clicks_gb.size()


def extract_p2(f3):
    return f3.groupby('Session ID')['Counts'].mean()


def extract_p3(clicks_gb):
    return clicks_gb.apply(lambda x: x['Timestamp'].iloc[-1] - x['Timestamp'].iloc[0]).astype('timedelta64[ms]')


def extract_p4(clicks_gb):
    return clicks_gb.apply(lambda x: x['Timestamp'].diff().astype('timedelta64[ms]').fillna(np.float64(0)).max())


def extract_p5(p1, p3):
    return p3.divide(p1 - 1).replace([np.inf, -np.inf], 0.0)


def extract_p6(f3):
    return f3.groupby('Session ID')['Counts'].max()


def extract_p10(f6):
    return f6.groupby('Session ID')['Sequent Clicks'].max()


def extract_p11(f7):
    int64_max = np.iinfo(np.int64).max

    return f7.groupby('Session ID')['Time Difference']\
        .apply(lambda group: group[group < int64_max].max())\
        .fillna(int64_max)


def extract_buys(clicks_group_keys, buys_group_keys):
    session_series = pd.Series(0, dtype=np.int8, index=clicks_group_keys)
    session_series[buys_group_keys] = np.int8(1)

    return np.array(session_series)
