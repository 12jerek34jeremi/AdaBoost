import numpy as np
import pandas as pd
from Stump import Stump
import functions
import gini


class AdaBoost():
    def __init__(self, data, metadata):  # done
        self.data = data
        self.metadata = metadata

        self.feature_n = data.shape[1] - 1
        self.rows_n = data.shape[0]
        self.stumps = []
        self.features_names = data.columns.to_list()[:-1]


        self.data["example_weight"] = pd.Series(np.full(self.rows_n, 1 / self.rows_n))
        self.output = self.data["output"].to_numpy()
        self.examples_weights = self.data["example_weight"].to_numpy()


    def add_new_stump(self):  # done
        new_stump = self.find_best_feature()
        correct_array = None
        feature = self.data[new_stump.feature_name].to_numpy()

        if new_stump.data_type == 'bool':
            correct_array = (feature == self.output)
        elif new_stump.data_type == 'con':
            correct_array = ((feature > new_stump.threshold) == self.output)
        elif new_stump.data_type == 'cat':
            correct_array = (np.isin(feature, new_stump.subset) == self.output)

        new_stump.total_error = np.sum(self.examples_weights[~correct_array])

        if new_stump.total_error == 1 or new_stump.total_error == 0:
            print("New Stump error is equal 0 or 1!")
            new_stump.pretty_print()
            return False

        new_stump.about_to_say = functions.amount_of_say(new_stump.total_error)

        factor = np.where(correct_array, -1.0, 1.0)
        self.data["example_weight"] *= np.exp(factor * new_stump.about_to_say)
        weights_sum = np.sum(self.examples_weights)
        self.data["example_weight"] /= weights_sum
        self.stumps.append(new_stump)

        return True

    def find_best_feature(self):  # done
        best_mean = None
        best_true_subset = None
        best_feature_name = None
        best_gini_index = 5.0  # all gini indices are less than or equal one

        for feature_name in self.features_names:

            if self.metadata[feature_name].data_type == "bool":
                gini_index = gini.calculate_gini_index_for_bool(
                    self.data[feature_name].to_numpy(), self.output, self.examples_weights)
                if gini_index < best_gini_index:
                    best_gini_index = gini_index
                    best_feature_name = feature_name

            elif self.metadata[feature_name].data_type == "con":
                gini_index, mean = gini.calculate_gini_index_for_continuous(
                    self.data[feature_name].to_numpy(), self.output, self.examples_weights)
                if gini_index < best_gini_index:
                    best_gini_index = gini_index
                    best_feature_name = feature_name
                    best_mean = mean

            elif self.metadata[feature_name].data_type == "cat":
                gini_index, true_subset = gini.calculate_gini_index_for_categorical(
                    self.data[feature_name].to_numpy(), self.output, self.examples_weights,
                    self.metadata[feature_name].true_subsets)
                if gini_index < best_gini_index:
                    best_gini_index = gini_index
                    best_feature_name = feature_name
                    best_true_subset = true_subset

        new_stump = Stump(best_feature_name, best_gini_index, self.metadata[best_feature_name].data_type)

        if new_stump.data_type == "con":
            new_stump.threshold = best_mean
        elif new_stump.data_type == "cat":
            new_stump.subset = best_true_subset

        return new_stump



    def train(self, stumps_nr):  # done
        self.stumps = []
        for i in range(stumps_nr):
            if not self.add_new_stump():
                return

    def predict(self, data_frame=None, activation=None):  # done
        if data_frame is None:
            data_frame = self.data

        output_array = np.zeros(data_frame.shape[0])

        for stump in self.stumps:
            output_array += stump.say(data_frame)

        if activation is not None:
            return activation(output_array)
        else:
            return output_array

    def predict_pretty_print(self, features, activation=None):
        """Features her is any object which can be indexed by string containing element which
                index is this stump feature name. For example dictionary or panda series"""
        my_sum = 0
        for stump, i in enumerate(self.stumps):
            print("Stump nr ", i)
            my_sum += stump.say_pretty_print(features)
            print("Sum is now: ", my_sum, "\n\n")

        if activation is not None:
            my_sum = activation(my_sum)

        print("Outputs is: ", my_sum)
        return my_sum

    def print(self):
        for i, stump in enumerate(self.stumps):
            print("Stump ", i)
            stump.pretty_print()
            print('\n')

    def test(self, data_frame=None):
        if data_frame is None:
            data_frame = self.data

        predictions = self.predict(data_frame) >= 0.0
        output = data_frame["output"].to_numpy() == 1.0

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
