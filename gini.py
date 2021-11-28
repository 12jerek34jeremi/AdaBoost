import numpy as np

def calculate_gini_index_for_bool(feature, output, examples_weights = None):
    if examples_weights is None:
        yes_to_yes = np.count_nonzero(np.logical_and(feature, output))
        yes_to_no = np.count_nonzero(np.logical_and(feature, ~output))
        no_to_no = np.count_nonzero(np.logical_and(~feature, ~output))
        no_to_yes = np.count_nonzero(np.logical_and(~feature, output))
    else:
        yes_to_yes = np.sum(examples_weights[np.logical_and(feature, output)])
        yes_to_no = np.sum(examples_weights[np.logical_and(feature, ~output)])
        no_to_no = np.sum(examples_weights[np.logical_and(~feature, ~output)])
        no_to_yes = np.sum(examples_weights[np.logical_and(~feature, output)])

    yes_count = yes_to_yes + yes_to_no
    no_count = no_to_no + no_to_yes

    if yes_count==0 or no_count==0:
        return 1.0

    gini_index1 = 1 - ((yes_to_yes / yes_count) ** 2) - ((yes_to_no / yes_count) ** 2)
    gini_index2 = 1 - ((no_to_no / no_count) ** 2) - ((no_to_yes / no_count) ** 2)
    gini_index = ((yes_count * gini_index1) + (no_count * gini_index2))/(yes_count+no_count)

    return gini_index


def calculate_gini_index_for_continuous(feature, output, examples_weights=None):
    means = np.unique(feature)
    means = (means[:-1] + means[1:]) / 2.0
    means = np.nan_to_num(means, nan=np.finfo(means.dtype).max)
    feature = np.nan_to_num(feature, nan=np.inf)

    gini_indices = [calculate_gini_index_for_bool(feature > mean, output, examples_weights) for mean in means]
    best_index = np.argmin(gini_indices)
    return gini_indices[best_index], means[best_index]


def calculate_gini_index_for_categorical(feature, output, real_subsets, examples_weights=None):
    gini_indices = [calculate_gini_index_for_bool(np.isin(feature, real_subset), output, examples_weights)
                    for real_subset in real_subsets]
    best_index = np.argmin(gini_indices)
    return gini_indices[best_index], real_subsets[best_index]