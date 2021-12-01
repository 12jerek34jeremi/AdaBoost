import pandas as pd
import pickle
from multiclass_adaboost import MulticlassAdaBoost


df = pd.read_csv("data/modified_golem_train.csv", index_col='Id')
metadata = None
with open("data/golem_metadata.pickle", "rb") as file:
    metadata = pickle.load(file)
pd.set_option("max_rows", None)

train_fraction = 0.7
train_rows_n = int(df.shape[0] * train_fraction)
df = df.sample(n=df.shape[0])
df_train = df.iloc[0:train_rows_n, :].copy()
df_test = df.iloc[train_rows_n:, :].copy()
stumps_nr, adab_nr = 7, 10









