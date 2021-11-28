import numpy as np
import pandas as pd
from AdaBoost import AdaBoost
import data as data_manager
import matplotlib.pyplot as plt
import pickle
from os import system
import pretty_print as pp


def clear():
    system('cls')

df = pd.read_csv("data/modified_golem_train.csv", index_col='Id')
metadata = None
with open("data/golem_metadata.pickle", "rb") as file:
    metadata = pickle.load(file)
cd 
