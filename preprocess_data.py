__author__ = 'pdanilov'

import random
import numpy as np


def slice_data(clicks, buys, frac=1):
    group_keys = clicks.groupby('Session ID').groups.keys()
    sample = random.sample(group_keys, int(frac * len(group_keys)))
    clicks.set_index('Session ID', drop=False, inplace=True)
    clicks = clicks.loc[sample]
    clicks.reset_index(drop=True, inplace=True)
    buys.set_index('Session ID', drop=False, inplace=True)
    buys = buys.loc[sample]
    buys.reset_index(drop=True, inplace=True)
    return [clicks, buys]


def df_group_by(df, sort=False):
    if sort:
        df.sort('Timestamp', inplace=True)
    df = df.groupby('Session ID')
    df_group_keys = np.array(list(df.groups.keys()), dtype=np.int32)
    return [df, df_group_keys]