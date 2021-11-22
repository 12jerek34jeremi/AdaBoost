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


def prepare_heart_data():
    with open("data/heart.csv", "rt") as file:
        data = pd.read_csv(file)

    data_types = ["con", "bool", "cat", "con", "con", "bool", "cat", "con", "bool", "con", "cat", "cat", "cat"]
    # con is continuous
    # bool is boolean
    # con is categorical

    metadata = {}
    for data_type, column_name in zip(data_types, data.columns):
        if data_type == "cat":
            metadata[column_name] = Metadata(data_type, true_subsets=generate_real_subsets(
                data[column_name].value_counts().index.to_list()))
        else:
            metadata[column_name] = Metadata(data_type)

    pd.set_option('max_columns', None)
    columns = ["sex", "fbs", "exng", "output"]
    data[columns] = data[columns].astype(bool)

    return data, metadata


def prepare_simple_data():
    chest_pain = [True, False, True, True, False, False, True, True]
    weight = [205, 180, 210, 167, 156, 125, 168, 172]
    output = [True, True, True, True, False, False, False, False]
    data_frame = pd.DataFrame({"chest_pain": chest_pain, "weight": weight, "output": output})
    metadata = {"chest_pain": Metadata("bool"), "weight": Metadata("con")}
    return data_frame, metadata


def prepare_heart_data_small() -> (pd.DataFrame, dict):
    data, metadata = prepare_heart_data()

    # indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175]
    indices = [0, 1, 3, 4, 6, 9, 165, 166, 168, 169, 172, 175]
    columns = ["age", "sex", "cp", "chol", "fbs", "output"]

    new_metadata = {}
    for column_name in metadata:
        if column_name in columns:
            new_metadata[column_name] = metadata[column_name]

    pd.set_option('max_columns', None)
    new_data = data.loc[indices, columns].copy()
    new_data.reset_index(drop=True, inplace=True)


    return new_data, new_metadata
