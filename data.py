import pickle
import pretty_print
import numpy as np
import pandas as pd
from Metadata import Metadata


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

