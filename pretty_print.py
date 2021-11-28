def print_list(my_list: list):
    for i, value in enumerate(my_list):
        print(i, ":", value)

def print_dict(my_dict: dict):
    for key, value in my_dict.items():
        print(key, ':', value)