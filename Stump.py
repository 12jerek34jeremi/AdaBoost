import numpy as np


class Stump:
    def __init__(self, feature_name, gini_index, data_type, about_to_say=0, total_error=0.0,
                 *, threshold=None, subset=None): #done

        self.feature_name = feature_name
        self.gini_index = gini_index
        self.data_type = data_type
        self.about_to_say = about_to_say
        self.total_error = total_error
        self.threshold = threshold
        self.subset = subset

    def say_pretty_print(self, features): #done
        """Features her is any object which can be indexed by string containing element which
        index is this stump feature name. For example dictionary or panda series"""

        feature = features[self.feature_name]
        if self.data_type == "bool":
            print(self.feature_name + "is True")
            if (feature == True):
                print("sum+= 1 * (", self.about_to_say, ")")
                return self.about_to_say
            else:
                print("sum+= (-1) * (", self.about_to_say, ")")
                return -self.about_to_say
        elif self.data_type == "con":
            print(self.feature_name + " > " + self.threshold)
            if (feature > self.threshold):
                print("sum+= 1 * (", self.about_to_say, ")")
                return self.about_to_say
            else:
                print("sum+= (-1) * (", self.about_to_say, ")")
                return -self.about_to_say
        elif self.data_type == "cat":
            print(self.feature_name, "belongs to ", self.subset)
            if (feature in self.subset):
                print("sum+= 1 * (", self.about_to_say, ")")
                return self.about_to_say
            else:
                print("sum+= (-1) * (", self.about_to_say, ")")
                return -self.about_to_say

    def say(self, data_frame): #done
        return_array = np.full(data_frame.shape[0], self.about_to_say)
        feature = data_frame[self.feature_name].to_numpy()
        if self.data_type == "bool":
            return_array[~feature] *= (-1)
        elif self.data_type == "con":
            return_array[feature <= self.threshold] *= (-1)
        elif self.data_type == "cat":
            return_array[feature not in self.subset] *= (-1)
        return return_array

    def pretty_print(self): #done
        if self.data_type == "bool":
            print(self.feature_name, "is True")
        elif self.data_type == "con":
            print(self.feature_name, " > ", self.threshold)
        elif self.data_type == "cat":
            print(self.feature_name, "belongs to ", self.subset)
        print("my about_to_say: ", self.about_to_say)
        print("my gini_index : ", self.gini_index)
