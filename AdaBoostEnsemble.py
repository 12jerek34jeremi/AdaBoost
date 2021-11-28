from AdaBoost import AdaBoost
from Metadata import Metadata
import pandas as pd
import numpy as np


class AdaBoostEnsemble:
    def __init__(self, ad_num:int, metadata: Metadata):
        self.adaboosts = []
        self.ad_num = ad_num
        self.metadata = metadata

    def train(self, data: pd.DataFrame, stump_nr:int):
        for i in range(stump_nr):
            new_adaboost = AdaBoost(self.metadata)
            new_adaboost.train(stump_nr, data)
            self.adaboosts.append(new_adaboost)

    def predict(self, df: pd.DataFrame, activation: callable):
        prediction = np.zeros(pd.shape[0])
        for adaboost in self.adaboosts:
            prediction += adaboost.predict(df)
        prediction /= self.ad_num

        if activation is None:
            return prediction
        else
            return activation(prediction)

