from os import system


def print_list(my_list: list):
    for i, value in enumerate(my_list):
        print(i, ":", value)


def print_dict(my_dict: dict, end='\n'):
    for key, value in my_dict.items():
        if type(value) == float:
            print(key, ':%.4f'%value, end=end)
        else:
            print(key, ':', value, end=end)


def clear():
    system('cls')
