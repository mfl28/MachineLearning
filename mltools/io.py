"""
Contains useful functions for loading data from files.
"""

from datetime import datetime
from pickle import HIGHEST_PROTOCOL, dump, load as pickle_load

from pandas import read_csv
from torch import load, save


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


def save_run(model, runs_dir, data=None):
    """
    Save a Pytorch model's state-dict and optionally additional data, such as
    train/test datasets for a run in a folder whose name contains the current date and
    time.
    :param model: (torch.nn.module) the module whose state-dict should be saved
    :param runs_dir: the root folder of the runs
    :param data: (optional, object)  additional data to save (with pickle)
    :return: the path of the folder containing the saved model state-dict and (optional) data
    """
    run_path = runs_dir / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"

    if not run_path.is_dir():
        run_path.mkdir(parents=True)

    model_file = run_path / "model.pkt"

    save(model.state_dict(), model_file)
    print(f"Successfully saved model to {model_file.resolve()}.")

    if data is not None:
        data_file = run_path / "data.pkl"
        with open(data_file, 'wb') as output_file:
            dump(data, output_file, HIGHEST_PROTOCOL)
        print(f"Successfully saved data to {data_file.resolve()}.")

    return run_path


def load_run(model, run_path):
    """
    Load a saved model and (if existing) additional data from disk.
    :param model: the model to be updated with the saved state-dict
    :param run_path: the path to the folder containing the files to load
    :return: the model and, optionally, additional data
    """
    if not run_path.is_dir():
        raise NotADirectoryError(f"Invalid run-directory path.")

    model_files = [file for file in run_path.iterdir() if file.suffix == ".pkt"]
    data_files = [file for file in run_path.iterdir() if file.suffix == ".pkl"]

    if len(model_files) == 0:
        raise ValueError(f"There are no model files to load in {run_path.name}.")

    if len(model_files) > 1 or len(data_files) > 1:
        raise ValueError(f"There exists more than one model- and/or data file in {run_path.name}.")

    model.load_state_dict(load(model_files[0]))
    print(f"Successfully loaded model data from {model_files[0].name}")

    if data_files:
        with open(data_files[0], 'rb') as data_file:
            data = pickle_load(data_file)
        print(f"Successfully loaded data from {data_files[0].name}")
        return model, data

    return model




