import data as data_manager
import pandas as pd
from AdaBoost import AdaBoost
from Metadata import Metadata


metrics_columns = ['accuracy', 'true_positive_rate', 'true_negative_rate', 'precision_positive', 'precision_negative', 'f1',
               'true_positive', 'true_negative', 'false_positive', 'false_negative']

def choose_stumps_nr(metadata: Metadata, iterator, df_train, df_test, print=True):
    columns = ['accuracy', 'true_positive_rate', 'true_negative_rate', 'precision_positive', 'precision_negative', 'f1',
               'true_positive', 'true_negative', 'false_positive', 'false_negative']
    metrics = []

    for stumps_nr in iterator:
        my_adaboost = AdaBoost(metadata)
        my_adaboost.train(stumps_nr, df_train.copy())
        metrics1 = my_adaboost.test(df_train)
        metrics2 = my_adaboost.test(df_test)
        metrics.append(pd.Series(metrics1, name=str(stumps_nr) +"st train"))
        metrics.append(pd.Series(metrics2, name=str(stumps_nr) + "st test"))
        if print:
            my_adaboost.print()

    return pd.DataFrame(metrics, columns=columns)


def different_values(df: pd.DataFrame, metadata: Metadata, iterator, train_fraction: float):
    metrics_value0 = choose_stumps_nr(df, metadata, iterator, train_fraction, 0)
    metrics_value1 = choose_stumps_nr(df, metadata, iterator, train_fraction, 1)
    metrics_value2 = choose_stumps_nr(df, metadata, iterator, train_fraction, 2)
    return metrics_value0, metrics_value1, metrics_value2