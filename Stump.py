import numpy as np


class Stump:
    def __init__(self ,feature_name, gini_index, feature_index, data_type, about_to_say=0, total_error=0.0,
                 *, mean=None, real_subset=None):
        self.feature_name = feature_name
        self.gini_index = gini_index
        self.data_type = data_type
        self.about_to_say = about_to_say
        self.total_error = total_error
        if(mean):
            self.mean = mean
        if(real_subset):
            self.real_subset = real_subset

    def amount_of_say(error):
        return 0.5 * np.log((1 - error) / error)

    def say(self, feature):
        if(self.data_type=="bool"):
            if(feature == True):
                return self.about_to_say
            else:
                return  -self.about_to_says
        elif(self.data_type == "con"):
            if(feature > self.mean):
                return self.about_to_say
            else:
                return -self.about_to_say
        elif(self.data_type == "cat"):
            if(feature in self.real_subset):
                return self.about_to_say
            else:
                return  -self.about_to_say

    def pretty_print(self, feature):
        if self.data_type == "bool":
            print(self.feature_name + "is True")
            if (feature == True):
                print("sum+= 1 * (", self.about_to_say, ")")
            else:
                print("sum+= (-1) * (", self.about_to_say, ")")
        elif self.data_type == "con":
            print(self.feature_name + " > " + self.mean)
            if (feature > self.mean):
                print("sum+= 1 * (", self.about_to_say, ")")
            else:
                print("sum+= (-1) * (", self.about_to_say, ")")
        elif self.data_type == "cat":
            print(self.feature_name, "belongs to ", self.real_subset)
            if (feature in self.real_subset):
                print("sum+= 1 * (", self.about_to_say, ")")
            else:
                print("sum+= (-1) * (", self.about_to_say, ")")

    def say_column(self, feature):
        return_array = np.full_like(feature, self.about_to_say)
        if self.data_type == "bool":
            return_array[feature==False] *= (-1)
        elif self.data_type == "con":
            return_array[feature<=self.mean] *= (-1)
        elif self.data_type == "cat":
            return_array[feature not in self.real_subset] *= (-1)
