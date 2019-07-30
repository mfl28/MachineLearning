"""
Contains useful functions for loading data from files.
"""

from pandas import read_csv


def get_data_from_csvs(train_data_path, test_data_path=None, **kwargs):
    """
    Reads data from the csv-files into pandas dataframes.
    :param train_data_path: Path to the training-data .csv file.
    :param test_data_path: (optional) Path to the test-data .csv file.
    :param kwargs: Keyword arguments passed to panda's read_csv.
    :return: pandas dataframe(s)
    """
    df_raw_train = read_csv(train_data_path, **kwargs)

    if test_data_path is not None:
        df_raw_test = read_csv(test_data_path, **kwargs)
        return df_raw_train, df_raw_test

    return df_raw_train
