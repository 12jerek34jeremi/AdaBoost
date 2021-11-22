import numpy as np

def calculate_gini_index_for_bool(feature, output, examples_weights):
    yes_to_yes = np.sum(examples_weights[np.logical_and(feature, output)])
    yes_to_no = np.sum(examples_weights[np.logical_and(feature, ~output)])
    no_to_no = np.sum(examples_weights[np.logical_and(~feature, ~output)])
    no_to_yes = np.sum(examples_weights[np.logical_and(~feature, output)])

    yes_count = yes_to_yes + yes_to_no
    no_count = no_to_no + no_to_yes

    gini_index1 = 1 - ((yes_to_yes / yes_count) ** 2) - ((yes_to_no / yes_count) ** 2)
    gini_index2 = 1 - ((no_to_no / no_count) ** 2) - ((no_to_yes / no_count) ** 2)
    gini_index = ((yes_count * gini_index1) + (no_count * gini_index2))

    return gini_index


def calculate_gini_index_for_continuous(feature, output, examples_weights):
    indices = np.argsort(feature)

    feature = feature[indices]
    output = output[indices]
    examples_weights = examples_weights[indices]

    means = (feature[:-1] + feature[1:]) / 2
    min_value, max_value = np.amin(feature), np.amax(feature)
    means = means[np.logical_and(means > min_value, means < max_value)]

    gini_indices = [calculate_gini_index_for_bool(feature > mean, output, examples_weights) for mean in means]
    best_index = np.argmin(gini_indices)
    return gini_indices[best_index], means[best_index]


def calculate_gini_index_for_categorical(feature, output, examples_weights, real_subsets):  # done
    gini_indices = [calculate_gini_index_for_bool(np.isin(feature, real_subset), output, examples_weights)
                    for real_subset in real_subsets]
    best_index = np.argmin(gini_indices)
    return gini_indices[best_index], real_subsets[best_index]