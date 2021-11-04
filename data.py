import pandas as pd

def prepere_heart_data():
    with open("data/heart.csv", "rt") as file:
        data = pd.read_csv(file)

    pd.set_option('max_columns', None)
    columns = ["sex", "fbs", "exng", "output"]
    data[columns] = data[columns].astype(bool)
    data_types = ["con", "bool", "cat", "con", "con", "bool", "cat", "con", "bool", "con", "cat", "cat", "cat"]
    # con is continuous
    # bool is boolean
    # con is categorical

    return data, data_types

def prepere_simple_data():
    chest_pain = [True, False, True, True, False, False, True, True]
    output = [True, True, True, True, False, False, False, False]
    weight = [205, 180, 210, 167, 156, 125, 168, 172]
    data_frame = pd.DataFrame({"chest_pain": chest_pain, "weight": weight, "output": output})
    data_types = ["bool", "con"]
    return data_frame, data_types