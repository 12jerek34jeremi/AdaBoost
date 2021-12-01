from AdaBoost import AdaBoost
from Metadata import Metadata
import pandas as pd
import numpy as np


class AdaBoostEnsemble:
    def __init__(self, adab_num: int, metadata: Metadata, true_value: int):
        self.adaboosts = []
        self.adab_num = adab_num
        self.metadata = metadata
        self.true_value = true_value

    def train(self, data: pd.DataFrame, stump_nr: int, train_on: int = 2000, extra_data_prepare: bool = True,
              equal: bool = False):
        weights = None
        if extra_data_prepare:
            if equal:
                values = data['output'].value_counts(dropna=False).index.to_numpy()
                weights_dict = dict(
                    zip(values, np.where(values == self.true_value, 0.5, 0.5 / (values.shape[0] - 1.0))))
            else:
                values = data['output'].value_counts(dropna=False).drop(self.true_value)
                sum = values.sum()
                values /= (2 * sum)
                weights_dict = values.to_dict()
                weights_dict[self.true_value] = 0.5
            weights = data['output'].replace(weights_dict)
        data['output'] = (data['output'] == self.true_value)

        for i in range(stump_nr):
            if weights is None:
                train_df = data.sample(n=train_on).copy()
            else:
                train_df = data.sample(n=train_on, weights=weights).copy()

            new_adaboost = AdaBoost(self.metadata)
            new_adaboost.train(stump_nr, train_df)
            self.adaboosts.append(new_adaboost)

    def predict(self, df: pd.DataFrame, activation: callable = None):
        prediction = np.zeros(df.shape[0])
        for adaboost in self.adaboosts:
            prediction += adaboost.predict(df)
        prediction /= self.adab_num

        if activation is None:
            return prediction
        else:
            return activation(prediction)

    def test(self, data_frame=None):

        predictions = self.predict(data_frame) >= 0.0
        output = data_frame["output"].to_numpy()

        metrics = {}
        true_positive = np.count_nonzero(np.logical_and(predictions, output))
        true_negative = np.count_nonzero(np.logical_and(~predictions, ~output))
        false_positive = np.count_nonzero(np.logical_and(predictions, ~output))
        false_negative = np.count_nonzero(np.logical_and(~predictions, output))
        all_count = output.shape[0]

        metrics["accuracy"] = (true_positive + true_negative) / all_count
        metrics["true_positive_rate"] = true_positive / (true_positive + false_negative)
        metrics["true_negative_rate"] = true_negative / (true_negative + false_positive)
        metrics["precision_positive"] = true_positive / (true_positive + false_positive)
        metrics["precision_negative"] = true_negative / (true_negative + false_negative)
        metrics["f1"] = (2 * metrics["true_positive_rate"] * metrics["precision_positive"]) / \
                        (metrics["true_positive_rate"] + metrics["precision_positive"])

        metrics["true_positive"] = true_positive
        metrics["true_negative"] = true_negative
        metrics["false_positive"] = false_positive
        metrics["false_negative"] = false_negative

        return metrics
