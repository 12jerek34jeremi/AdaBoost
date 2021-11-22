import numpy as np
import pandas as pd
from AdaBoost import AdaBoost
import data as data_manager
import matplotlib.pyplot as plt

data, meta_data = data_manager.prepare_heart_data_small()
my_adaboost = AdaBoost(data, meta_data)
my_adaboost.train(10)
