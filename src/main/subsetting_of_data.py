import sys
import os
from read_and_write_data import read_clicks, read_buys, write_df
from preprocess_data import slice_data


if __name__ == "__main__":
    assert len(sys.argv) == 5, r"Incorrect argument list, names also shouldn't contain spaces" + \
                               "\nusage: argv[0] /dir/with/data clicks_file buys_file frac"

    path = sys.argv[1]
    file_clicks_basename = sys.argv[2]
    file_buys_basename = sys.argv[3]
    char_frac = sys.argv[4]
    float_frac = float(char_frac)

    assert float_frac > 0, 'Frac must be a positive number'

    frac = float_frac if float_frac <= 1 else int(float_frac)
    if frac == 1:
        exit(0)

    file_clicks_parts = os.path.splitext(file_clicks_basename)
    file_buys_parts = os.path.splitext(file_buys_basename)
    file_clicks_sliced_basename = file_clicks_parts[0] + '-' + char_frac + file_clicks_parts[1]
    file_buys_sliced_basename = file_buys_parts[0] + '-' + char_frac + file_buys_parts[1]
    file_clicks = os.path.join(path, file_clicks_basename)
    file_buys = os.path.join(path, file_buys_basename)
    file_clicks_sliced = os.path.join(path, file_clicks_sliced_basename)
    file_buys_sliced = os.path.join(path, file_buys_sliced_basename)

    clicks = read_clicks(file_clicks)
    buys = read_buys(file_buys)
    clicks, buys = slice_data(clicks, buys, frac=frac)

    write_df(clicks, file_clicks_sliced)
    write_df(buys, file_buys_sliced)