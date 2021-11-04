import numpy as np
import pandas as pd
from Stump import Stump


class AdaBoost():
    def __init__(self, data, data_types,  max_about_to_say = 10.0):
        self.data = data
        self.data_types = data_types
        self.feature_n = data.shape[1] - 1
        self.rows_n = data.shape[0]
        self.output = self.data.iloc[:, self.feature_n].to_numpy()
        self.real_subsets = [None for i in range(self.feature_n)]

        for i in range(self.feature_n):
            if (self.data_types[i] == "cat"):
                self.real_subsets[i] = self.generate_real_subsets(data.iloc[:, i].value_counts().index.to_list())

        self.data["example_weight"] = pd.Series(np.full(self.rows_n, 1 / self.rows_n))
        self.examples_weights = self.data["example_weight"].to_numpy()
        self.stumps = []

    def sort_me(self, by):
        self.data.sort_values(by=by)
        self.output = self.data.iloc[:, self.feature_n].to_numpy()
        self.examples_weights = self.data["example_weight"].to_numpy()


    def add_new_stump(self):
        new_stump = self.find_best_feature()
        ccorrect_incorrect_array = None

        feature = self.data[new_stump.feature_name].to_numpy()

        if new_stump.data_type == 'bool':
            correct_array = (feature == self.output)
        elif new_stump.data_type == 'con':
            correct_array = ((feature>new_stump.mean) == self.output)
        elif new_stump.data_type == 'cat':
            correct_array = (np.isin(feature, new_stump.real_subset) == self.output)

        new_stump.total_error = np.sum(self.examples_weights[~correct_array])

        if new_stump.total_error == 1 or new_stump.total_error == 0:
            print("New Stump error is equal 0 or 1!")
            return new_stump

        new_stump.about_to_say = self.amount_of_say(new_stump.total_error)

        self.examples_weights[~correct_array] *= np.exp(new_stump.about_to_say)
        self.examples_weights[correct_array] *= np.exp(-new_stump.about_to_say)
        self.stumps.append(new_stump)

        return None




    def find_best_feature(self):
        best_mean = None
        best_real_subset = None
        best_feature_index = -1  # all feature indices are non negative
        best_gini_index = 5.0  # all gini indices are less than or equal zero

        for i in range(self.feature_n):

            if (self.data_types[i] == "bool"):
                gini_index = self.calculate_gini_index_for_bool(self.data.iloc[:, i].to_numpy())
                if (gini_index < best_gini_index):
                    best_gini_index = gini_index
                    best_feature_index = 1

            elif (self.data_types[i] == "con"):
                gini_index, mean = self.calculate_gini_index_for_bool(self.data.iloc[:, i].to_numpy())
                self.sort_me(self.data.columns[i])
                if (gini_index < best_gini_index):
                    best_gini_index = gini_index
                    best_feature_index = 1
                    best_mean = mean

            elif (self.data_types[i] == "con"):
                gini_index, real_subset = self.calculate_gini_index_for_bool(self.data.iloc[:, i].to_numpy())
                if (gini_index < best_gini_index):
                    best_gini_index = gini_index
                    best_feature_index = 1
                    best_real_subset = real_subset

        return Stump(self.data.columns[i], best_gini_index, i, self.data_types[best_feature_index],
                                 mean=best_mean, real_subset=best_real_subset)

    def calculate_gini_index_for_bool(self, feature):
        yes_to_yes = np.sum(self.examples_weights[np.logical_and(feature, self.output)])
        yes_to_no = np.sum(self.examples_weights[np.logical_and(feature, ~self.output)])
        no_to_no = np.sum(self.examples_weights[np.logical_and(~feature, ~self.output)])
        no_to_yes = np.sum(self.examples_weights[np.logical_and(~feature, self.output)])

        yes_count = yes_to_yes + yes_to_no
        no_count = no_to_no + no_to_yes

        gini_index1 = 1 - (yes_to_yes / yes_count) ** 2 - (yes_to_no / yes_count) ** 2
        gini_index2 = 1 - (no_to_no / no_count) ** 2 - (no_to_yes / no_count) ** 2
        gini_index = (yes_count * gini_index1 + no_count * gini_index2) / (yes_count + no_count)

        return gini_index

    def calculate_gini_index_for_continuous(self, feature):
        means = (feature[:-1] + feature[1:]) / 2
        gini_indeces = [self.calculate_gini_index_for_bool(feature > mean) for mean in means]
        best_index = np.argmin(gini_indeces)
        return gini_indeces[best_index], means[best_index]

    def calculate_gini_index_for_categorical(self, feature, real_subsets):
        gini_indeces = [self.calculate_gini_index_for_bool(np.isin(feature, real_subset))
                        for real_subset in real_subsets]

        best_index = np.argmin(gini_indeces)
        return gini_indeces[best_index], real_subsets[best_index]

    def generate_real_subsets(self, set):
        n = len(set)
        height = (2 ** n) // 2
        zeros_ones_array = [[0 for i in range(n)] for i in range(height)]
        step = height
        for j in range(1, n):
            for i in range(0, height, step):
                for k in range(i + (step // 2), i + step, 1):
                    zeros_ones_array[k][j] = 1
            step = step // 2

        zeros_ones_array.pop(0)
        height -= 1

        real_subsets = [[] for i in range(height)]
        for subset, zeros_ones_row in zip(real_subsets, zeros_ones_array):
            for element, zero_one in zip(set, zeros_ones_row):
                if zero_one == 1:
                    subset.append(element)

        return real_subsets

    def pretty_print_list(self, my_list):
        for index, row in zip(range(len(my_list)), my_list):
            print(index, row)

    def amount_of_say(error):
        return 0.5 * np.log((1 - error) / error)

    def train(self, stumps_nr):
        self.stumps = []
        for i in range(stumps_nr):
            new_stump = self.add_new_stump()
            if self.add_new_stump() is not None:
                self.incorrect_stump = new_stump
                return 0

    def predict_one(self, feature_list):
        sum = 0
        for stump, feature in zip(self.stumps, feature_list):
            sum += stump.say_one(feature)
        if sum>=0:
            return True
        else:
            return False

    def predict_multi(self, data_frame):
        sum_array = np.full(data_frame.shape[1], 0.0)
        for stump in self.stumps:
            sum_array += stump.predict_multi(data_frame[stump.feature_name].to_numpy)

        data_frame["output"] = pd.Series(sum_array >= 0)

    def test(self, data_frame = None):
        if data_frame is None:
            data_frame = self.data

        output_array = np.full(data_frame.shape[1], 0.0)
        for stump in self.stumps:
            output_array += stump.predict_multi(data_frame[stump.feature_name].to_numpy)

        output_array = (output_array >= 0.0)
        correctly_classified = np.count_nonzero(output_array == data_frame["output"].to_numpy())
        incorrectly_clasified = data_frame.columns.shape[0] - correctly_classified

        return correctly_classified, incorrectly_clasified

    def predict_pretty_print(self, feature_list):
        sum = 0
        i = 0
        for stump, feature in zip(self.stumps, feature_list):
            sum += stump.say_one(feature)
            print("Stump nr ", i)
            stump.pretty_print(feature)
            print("Sum is now: ", sum, "\n\n")
            i += 1

        if sum >= 0:
            print("output value is True (sum>=0)")
        else:
            print("output value is False (sum>=0)")


