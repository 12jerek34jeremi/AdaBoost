import pickle
import pretty_print
import numpy as np
import pandas as pd
from Metadata import Metadata

metrics_columns = ['accuracy', 'true_positive_rate', 'true_negative_rate', 'precision_positive', 'precision_negative',
                   'f1', 'true_positive', 'true_negative', 'false_positive', 'false_negative']


def generate_real_subsets(set):
    n = len(set)
    height = (2 ** n) // 2
    zeros_ones_array = [[0 for i in range(n)] for i in range(height)]
    step = height
    for j in range(1, n):
        for i in range(0, height, step):
            for k in range(i + (step // 2), i + step, 1):
                zeros_ones_array[k][j] = 1
        step = step // 2

    zeros_ones_array.pop(0)
    height -= 1

    real_subsets = [[] for i in range(height)]
    for subset, zeros_ones_row in zip(real_subsets, zeros_ones_array):
        for element, zero_one in zip(set, zeros_ones_row):
            if zero_one == 1:
                subset.append(element)

    return real_subsets


def create_meta_data(data_types: dict[str, str], df: pd.DataFrame) -> dict[str: Metadata]:
    metadata = {}
    for column_name, data_type in data_types.items():
        if data_type == "cat":
            metadata[column_name] = Metadata(data_type, true_subsets=generate_real_subsets(
                df[column_name].value_counts().index.to_list()))
        else:
            metadata[column_name] = Metadata(data_type)
    return metadata


def prepare_data(data_file:str, metadata_file:str, index:str) -> (pd.DataFrame, Metadata):
    data = pd.read_csv(data_file, index=index)
    with open(metadata_file, 'rb') as file:
        metadata = pickle.load(file)
    return data, metadata


def create_data_sets(df: pd.DataFrame, train_fraction: float, true_value: int, equal: bool = False) -> (pd.DataFrame, pd.DataFrame):
    train_rows_n = int(df.shape[0] * train_fraction)
    df = df.sample(n=df.shape[0])
    df_train = df.iloc[0:train_rows_n, :]
    df_test = df.iloc[train_rows_n:, :].copy()
    df_test['output'] = (df_test['output'] == true_value)

    # create weights dict
    weights_dict = None
    if equal:
        values = df['output'].value_counts(dropna=False).index.to_numpy()
        weights_dict = dict(zip(values, np.where(values == true_value, 0.5, 0.5 / (values.shape[0]-1.0))))
    else:
        values = df['output'].value_counts(dropna=False).drop(true_value)
        sum = values.sum()
        values /= (2*sum)
        weights_dict = values.to_dict()
        weights_dict[true_value] = 0.5

    weights = df_train['output'].replace(weights_dict)
    train_values = df_train['output'].value_counts(dropna=False)
    n = train_values.min() * train_values.size
    df_train = df_train.sample(n=n, weights=weights).copy()
    df_train['output'] = (df_train['output'] == true_value)

    return df_train, df_test

