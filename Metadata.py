class Metadata:
    def __init__(self, data_type, *, true_subsets = None):
        self.data_type = data_type
        self.true_subsets = true_subsets

    def __str__(self):
        my_str =  "Metadata Object{data type is "+self.data_type
        if self.true_subsets:
            my_str += ", true_subsets is set.}"
        else:
            my_str += ", true_subsets is None.}"

        return my_str

    def __repr__(self):
        return self.__str__()
