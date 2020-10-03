import pandas as pd
import os
import sys

data_path = "data\iris.data"
column_names = ["sepal_length", "sepal_width",
                "petal_length", "petal_width", "class"]


def is_float_list(iterable):
    """Checks if all elements of an iterable are floats
    """
    for element in iterable:
        if type(element) is not float:
            return False
    return True


def ask_for_flower_data():
    """Asks for user input and processes it to the desired format

    Returns:

    flower_data: A list of 4 float values
    """
    flower_data = tuple(input(
        f"Enter the 4 values, separated by spaces and in order({column_names[:-1]}), you want to use to classify a flower\n").split())
    try:
        flower_data = tuple(float(item) for item in flower_data)
    except:
        print("Error during conversion. Please check input values")
        return ()
    return flower_data


def classify(data_iterable):
    """Given an interable of 4 features, classify into one of the classes

    Args:
        data_iterable: iterable of 4 features (not class) to use as input data
    """
    print(data_iterable)
    iris_df = pd.read_csv(data_path, names=column_names)
    print(iris_df)


if __name__ == "__main__":
    flower_data = ask_for_flower_data()
    while len(flower_data) != 4 or not is_float_list(flower_data):
        print("*WARNING: Invalid data. Try again*\n")
        flower_data = ask_for_flower_data()
    print('Correctly stored data. Proceeding to classification...')
    classify(flower_data)
