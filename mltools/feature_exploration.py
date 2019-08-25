"""
Contains useful functions for feature exploration.
"""

from matplotlib.pyplot import get_cmap
from pandas import DataFrame, Series


def get_feature_names_with_overrepresented_values(df, threshold):
    """
    This function returns a list of features where a single value is taken on
    by at least threshold*100 % of samples.
    :param df: (pandas.DataFrame) the dataframe to be examined
    :param threshold: (float) the ratio threshold
    :return: a list of overrepresented features
    """
    return [feature for feature in df.columns
            if any(perc >= threshold for perc in df[feature].value_counts() / df[feature].count())]


def plot_hists_of_overrepresented_values(df, threshold, max_len=None):
    """
    Shows a histogram of overrepresented values in a dataframe.
    :param df: (pandas.DataFrame) the dataframe to be examined
    :param threshold: (float) the ratio threshold
    :param max_len: (optional, int) maximum number of columns to plot
    :return: the histogram
    """
    cols_to_plot = get_feature_names_with_overrepresented_values(df, threshold)

    if max_len is not None and max_len < len(cols_to_plot):
        cols_to_plot = cols_to_plot[:max_len]

    return df.hist(column=cols_to_plot, bins=10, figsize=(15, 15))


def show_corr(df, target_name=None, method="pearson"):
    """
    Shows correlations of a column (feature) with respect to all other features.
    :param df: (pandas.DataFrame) the dataframe
    :param target_name: (string) the name of the target whose correlations with other features are sought
    :return: correlations
    """
    if target_name is not None:
        return DataFrame(df.corr(method=method)[target_name].sort_values(ascending=False)).style.format("{:.2}") \
            .background_gradient(cmap=get_cmap("coolwarm"))

    return df.corr(method=method).style.format("{:.2}").background_gradient(cmap=get_cmap("coolwarm"), axis=1)


def get_feature_types(df):
    """
    Gets all features in the dataframe grouped by whether they are numerical or non-numerical (e.g. string).
    :param df: (pandas.DataFrame) the dataframe
    :return: a dict containing the categorical and numerical features
    """
    cn_features = {
        'numerical': [f_name for f_name in df.columns if df[f_name].dtype != 'object'],
        'non_numerical': [f_name for f_name in df.columns if df[f_name].dtype == 'object']
    }
    return cn_features


def extended_info(df, description=None):
    """
    This function returns a dateframe containing extended information about the passed in
    dataframe object.
    :param df: (pandas.DataFrame) the dataframe to be analysed
    :param description: (optional, dict) a dictionary containing descriptions of the features as strings
    :return: a pandas.DataFrame containing the sought information
    """
    df_info = DataFrame(index=df.columns)
    df_info['type'] = df.dtypes
    df_info['non-null'] = df.notnull().sum()
    df_info['nunique'] = df.nunique()

    if description is not None:
        df_info['description'] = Series(description)
        df_info['description'].fillna("", inplace=True)

    return df_info
