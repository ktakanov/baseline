__author__ = 'pdanilov'

import random


def slice_data(clicks, buys, frac=1):
    clicks_grouped = clicks.groupby('Session ID')
    group_keys = clicks_grouped.groups.keys()
    sample = random.sample(group_keys, int(frac * len(group_keys)))
    clicks.set_index('Session ID', drop=False, inplace=True)
    clicks = clicks.loc[sample]
    clicks.reset_index(drop=True, inplace=True)
    buys.set_index('Session ID', drop=False, inplace=True)
    buys = buys.loc[sample]
    buys.reset_index(drop=True, inplace=True)


def sort_and_group_by(clicks, buys):
    clicks = clicks.sort('Timestamp')
    clicks = clicks.groupby('Session ID')
    buys = buys.groupby('Session ID')
    return [clicks, buys]