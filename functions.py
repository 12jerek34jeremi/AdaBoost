import numpy as np

def amount_of_say(error):
    return 0.5 * np.log((1 - error) / error)

def sigma(arg):
    return 1.0 / (1.0 + np.exp(-arg))

def perceptron(arg):
    if isinstance(arg, np.ndarray):
        return_array = np.zeros_like(arg)
        return_array[arg>=0.0] += 1.0
        return return_array

    elif isinstance(arg, float):
        if arg >= 0.0:
            return 1.0
        else:
            return 0.0
    else:
        raise Exception("Can't use perception function on type: " + type(arg))
        return 0.0
