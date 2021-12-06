import numpy as np
import pickle
from AdaBoostEnsemble import AdaBoostEnsemble
from Metadata import Metadata
import pandas as pd

class MulticlassAdaBoost:
    def __init__(self, adab_num: int, metadata: Metadata):
        self.ensembles = []
        self.adab_num = adab_num
        self.metadata = metadata
        self.true_values = []

    def train(self, data: pd.DataFrame, stump_nr: int, train_on: int = 2000, extra_data_prepare: bool = True,
              equal: bool = False):
        self.true_values = data['output'].value_counts().index.to_list()
        for true_value in self.true_values:
            new_ensemble = AdaBoostEnsemble(self.adab_num, self.metadata, true_value)
            new_ensemble.train(data.copy(), stump_nr, train_on, extra_data_prepare, equal)
            self.ensembles.append(new_ensemble)

    def predict(self, df: pd.DataFrame, save_to_df: bool = False):
        predictions = []
        for ensemble in self.ensembles:
            predictions.append(ensemble.predict(df))
        prediction = np.column_stack(predictions)
        prediction = np.argmax(prediction, axis=1)

        if save_to_df:
            df['prediction'] = pd.Series(prediction, index=df.index)
        return prediction

    def test(self, df: pd.DataFrame):
        prediction = self.predict(df)
        output = df['output'].to_numpy()
        correct_count = np.count_nonzero(prediction == output)
        return correct_count/output.shape[0]

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

