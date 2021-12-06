# import pandas as pd
# import pickle
# from multiclass_adaboost import MulticlassAdaBoost
# import numpy as np
#
# df_train = pd.read_csv("data/modified_golem_train.csv", index_col='Id')
# df_test = pd.read_csv("data/modified_golem_test.csv", index_col='Id')
# with open("data/golem_metadata.pickle", "rb") as file:
#     metadata = pickle.load(file)
#
# with open('data/my_adaboost_i190.pickle', 'rb') as file:
#     adaboosts, results = pickle.load(file)
#
# results = []
# train_rows = int(0.7*df_train.shape[0])
# df_train_test = df_train.iloc[train_rows:, :]
# best_adaboost = None
# best_result = 2.0
#
# for adaboost in adaboosts:
#     result = adaboost.test(df_train_test)
#     results.append(result)
#     if result < best_result:
#         best_result = result
#         best_adaboost = adaboost
#
#
# all_result = best_adaboost.test(df_train)
# train_result = best_adaboost.test(df_train.iloc[0:train_rows, :])
#
# with open('data/best_adaboost.pickle', 'wb') as file:
#     pickle.dump(best_adaboost, file)
#
# my_answer = best_adaboost.predict(df_test, True)
import pandas as pd

my_answer = pd.read_csv('data/my_answer.csv', index_col='Id')
print(my_answer.shape)
