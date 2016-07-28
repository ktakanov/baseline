import unittest
import pandas as pd
import numpy as np
from pandas.tslib import Timestamp
from src.main.feature_extraction import extract_f3, extract_f6, extract_f7, extract_p1, extract_p2, extract_p3,\
    extract_p4, extract_p5, extract_p6, extract_p10, extract_p11, no_cons_clicks_indicator


class TestFeatureExtraction(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.df = pd.DataFrame(
            {'Session ID': pd.Series([5, 5, 5, 10, 10, 15, 15, 15, 20, 20, 20, 20, 25, 25, 25, 25, 25, 25, 30, 30],
                                     dtype=np.int32),
             'Timestamp': pd.Series([
                 Timestamp('2014-04-07 17:13:46.713'),
                 Timestamp('2014-04-07 17:20:56.973'),
                 Timestamp('2014-04-07 17:21:19.602'),
                 Timestamp('2014-04-04 07:44:14.590'),
                 Timestamp('2014-04-04 07:45:20.245'),
                 Timestamp('2014-04-05 08:14:28.645'),
                 Timestamp('2014-04-05 08:15:49.200'),
                 Timestamp('2014-04-05 08:19:55.455'),
                 Timestamp('2014-04-03 14:42:25.879'),
                 Timestamp('2014-04-03 14:42:43.585'),
                 Timestamp('2014-04-03 14:44:40.147'),
                 Timestamp('2014-04-03 14:45:09.199'),
                 Timestamp('2014-04-07 03:32:19.078'),
                 Timestamp('2014-04-07 03:32:44.139'),
                 Timestamp('2014-04-07 03:32:47.525'),
                 Timestamp('2014-04-07 03:32:48.647'),
                 Timestamp('2014-04-07 03:32:57.321'),
                 Timestamp('2014-04-07 03:32:59.910'),
                 Timestamp('2014-04-06 11:11:19.232'),
                 Timestamp('2014-04-06 11:11:19.232')]),
             'Item ID': np.array([214530776, 214530776, 214530776, 214820942, 214826810, 214555903, 214547255,
                                  214547255, 214829282, 214718203, 214829282, 214819552, 214836761, 214839313,
                                  214839313, 214839313, 214839313, 214839313, 214820201, 214820201],
                                 dtype=np.int32)
             }
        )
        self.gb = self.df.groupby('Session ID')

    def test_f3(self):
        src = extract_f3(self.gb).sort_values(['Session ID', 'Item ID'])
        dst = pd.DataFrame(np.array([
            [5, 214530776, 3],
            [10, 214820942, 1],
            [10, 214826810, 1],
            [15, 214555903, 1],
            [15, 214547255, 2],
            [20, 214829282, 2],
            [20, 214718203, 1],
            [20, 214819552, 1],
            [25, 214836761, 1],
            [25, 214839313, 5],
            [30, 214820201, 2]],
            dtype=np.int32),
            columns=['Session ID', 'Item ID', 'Counts']
        ).sort_values(['Session ID', 'Item ID'])
        np.testing.assert_array_equal(src, dst, 'Incorrect extraction of F3')

    def test_f6(self):
        src = extract_f6(self.gb).sort_values(['Session ID', 'Item ID'])
        dst = pd.DataFrame(np.array([
            [5, 214530776, 3],
            [10, 214820942, 1],
            [10, 214826810, 1],
            [15, 214555903, 1],
            [15, 214547255, 2],
            [20, 214829282, 1],
            [20, 214718203, 1],
            [20, 214819552, 1],
            [25, 214836761, 1],
            [25, 214839313, 5],
            [30, 214820201, 2]],
            dtype=np.int32),
            columns=['Session ID', 'Item ID', 'Sequent Clicks']
        ).sort_values(['Session ID', 'Item ID'])
        np.testing.assert_array_equal(src, dst, 'Incorrect extraction of F6')

    def test_f7(self):
        src = extract_f7(self.gb).sort_values(['Session ID', 'Item ID'])
        dst = pd.DataFrame(np.array([
            [5, 214530776, 430260],
            [10, 214820942, no_cons_clicks_indicator],
            [10, 214826810, no_cons_clicks_indicator],
            [15, 214555903, no_cons_clicks_indicator],
            [15, 214547255, 246255],
            [20, 214829282, no_cons_clicks_indicator],
            [20, 214718203, no_cons_clicks_indicator],
            [20, 214819552, no_cons_clicks_indicator],
            [25, 214836761, no_cons_clicks_indicator],
            [25, 214839313, 8674],
            [30, 214820201, 0]]),
            columns=['Session ID', 'Item ID', 'Time Difference']
        ).sort_values(['Session ID', 'Item ID'])
        np.testing.assert_array_equal(src, dst, 'Incorrect extraction of F7')

    def test_p1(self):
        src = extract_p1(self.gb)
        dst = np.array([3, 2, 3, 4, 6, 2])
        np.testing.assert_array_equal(src, dst, 'Incorrect extraction of P1')

    def test_p2(self):
        src = extract_p2(extract_f3(self.gb))
        dst = np.array([3, 1, 1.5, 4.0 / 3.0, 3, 2])
        np.testing.assert_array_equal(src, dst, 'Incorrect extraction of P2')

    def test_p3(self):
        src = extract_p3(self.gb)
        dst = np.array([452889.0, 65655.0, 326810.0, 163320.0, 40832.0, 0.0])
        np.testing.assert_array_almost_equal(src, dst, decimal=-3, err_msg='Incorrect extraction of P3')

    def test_p4(self):
        src = extract_p4(self.gb)
        dst = np.array([430260.0, 65655.0, 246255.0, 116562.0, 25061.0, 0.0])
        np.testing.assert_array_almost_equal(src, dst, decimal=-3, err_msg='Incorrect extraction of P4')

    def test_p5(self):
        src = extract_p5(extract_p1(self.gb), extract_p3(self.gb))
        dst = np.array([226444.5, 65655.0, 163405.0, 54440.0, 8166.3999999999996, 0.0])
        np.testing.assert_array_almost_equal(src, dst, decimal=-3, err_msg='Incorrect extraction of P5')

    def test_p6(self):
        src = extract_p6(extract_f3(self.gb))
        dst = np.array([3, 1, 2, 2, 5, 2])
        np.testing.assert_array_equal(src, dst, 'Incorrect extraction of P6')

    def test_p10(self):
        src = extract_p10(extract_f6(self.gb))
        dst = np.array([3, 1, 2, 1, 5, 2])
        np.testing.assert_array_equal(src, dst, 'Incorrect extraction of P10')

    def test_p11(self):
        src = extract_p11(extract_f7(self.gb))
        dst = np.array([430260.0, no_cons_clicks_indicator, 246255.0, no_cons_clicks_indicator, 8674.0, 0.0])
        np.testing.assert_array_equal(src, dst, 'Incorrect extraction of P11')