import random
import numpy as np


def slice_data(clicks, buys, frac=1):
    clicks_group_keys = list(clicks.groupby('Session ID').groups.keys())
    num_of_samples = frac if frac > 1 else int(frac * len(clicks_group_keys))
    clicks_sample = random.sample(clicks_group_keys, num_of_samples)
    clicks.set_index('Session ID', drop=False, inplace=True)
    clicks = clicks.loc[clicks_sample]
    clicks.reset_index(drop=True, inplace=True)
    buys_group_keys = list(buys.groupby('Session ID').groups.keys())
    buys_sample = list(set(buys_group_keys) & set(clicks_sample))
    buys.set_index('Session ID', drop=False, inplace=True)
    buys = buys.loc[buys_sample]
    buys.reset_index(drop=True, inplace=True)

    return clicks, buys


def df_group_by_session_id(df):
    df = df.groupby('Session ID')
    keys = sorted(list(df.groups.keys()))
    df_group_keys = np.array(keys, dtype=np.int32)

    return df, df_group_keys
