import pandas as pd
import pickle
import numpy as np
from Metadata import Metadata
import itertools
from os import system

data_types = None
# with open("data/data_types_golem_train.pickle", "rb") as file2:
#     data_types = pickle.load(file2)
file = open("data/data_types_test_notes.txt", "at")
df = pd.read_csv("data/modified_golem_test.csv", index_col="Id")
df_original = pd.read_csv("data/golem_test.csv", index_col="Id")
column = None
p1, p2 = 0, 10
# pd.set_option("max_rows", None)


def categorise() -> dict[str, int]:
    global column
    global df
    categories_list = list(map(str, df[column].value_counts(dropna=False).index.to_list()))
    categories_list.sort()
    categories = dict(zip(categories_list, range(len(categories_list))))
    df[column] = df[column].replace(categories)

    return categories


def note_data_type(data_type: str, true_value:str = None, false_value:str = None, dictionary=None) -> None:
    global file
    global df
    global data_types
    global column
    my_str = None
    if data_type == "con":
        file.write("The '" + column + "' is continuous data.\n\n")
    elif data_type == "bool":
        categories = {true_value: True, false_value: False}
        replace_with_dict(categories)
        df = df.astype({column: 'bool'})
        file.write("The '" + column + "' is boolean data.\n" +
                   true_value + " is interpreted as True.\n" +
                   false_value + " is interpreted as False.\n\n")
    elif data_type == "cat":
        if dictionary is None:
            my_str = str(categorise())
        else:
            df[column] = df[column].replace(dictionary)
            my_str = str(dictionary)
        file.write("The '" + column + "' is categorical data.\n"
                    "Each categorie was assigned to a categorie index as seen below.\n"
                    "Then values of columns have been replaced by categories indices.\n" + my_str + "\n\n")
    # data_types[column] = data_type
    file.flush()
    return my_str


def save():
    global df
    global data_types
    # with open("data/data_types_golem_train.pickle", "wb") as file2:
    #     pickle.dump(data_types, file2)
    df.to_csv("data/modified_golem_test.csv")


def create_dict(first_value: str, firs_new_value: int, step: int = 1) -> dict[str, str]:
    global df
    global column
    value_list = list(map(str, df[column].value_counts(dropna=False).index.to_list()))
    value_list.sort()
    return_dictionary = {}
    index = value_list.index(first_value)
    for value, new_value in zip(value_list[index:], itertools.count(firs_new_value, step)):
        return_dictionary[value] = str(new_value)
    return return_dictionary


def replace_with_dict(dictionary: dict) -> None:
    global df
    global column
    df[column] = df[column].replace(dictionary)


def values():
    global df
    global column
    return df[column].value_counts(dropna=False).sort_index()


def iloc():
    global df
    global column
    global p1
    global p2
    return df.iloc[0:10, p1:p2]


def to_float():
    global df
    global column
    df = df.astype({column: 'float64'})


def replace(value: str, new_value: str) -> None:
    global df
    global column
    df[column] = df[column].replace({value: new_value})


def clear():
    system('cls')


def col():
    global column
    print(column)


def nc(): # next column
    global df
    global column
    columns = df.columns.to_list()
    column = columns[columns.index(column) + 1]
    return column
