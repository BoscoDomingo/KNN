import pandas as pd
import numpy as np
from collections import Counter
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
        f"\nEnter the 4 values, separated by spaces and in order({column_names[:-1]}), you want to use to classify a flower\n").split())
    try:
        flower_data = tuple(float(item) for item in flower_data)
    except:
        print("Error during conversion. Please check input values")
        return ()
    return flower_data


def ask_for_neighbours():
    """Asks for user input for the number of neighbours
    """
    number_of_neighbours = input(
        "\nPlease indicate, with an integer, the number of neighbours desired\n")
    try:
        number_of_neighbours = int(number_of_neighbours)
    except:
        print("Error during conversion. Please check input value")
        return 0
    return number_of_neighbours


def get_data():
    flower_data = ask_for_flower_data()
    while len(flower_data) != 4 or not is_float_list(flower_data):
        print("*WARNING: Invalid data. Try again*\n")
        flower_data = ask_for_flower_data()
    number_of_neighbours = ask_for_neighbours()
    while number_of_neighbours <= 0 or type(number_of_neighbours) is not int:
        number_of_neighbours = ask_for_neighbours()
    return flower_data, number_of_neighbours


def distance_between_vectors(vector1, vector2):
    # return np.sqrt(np.sum([x**2 for x in (vector1 - vector2)]))
    return np.sqrt(np.sum([x**2 for x in np.subtract(vector1, vector2)]))


def knn_classify(data_iterable, number_of_neighbours):
    """Given an iterable of 4 features and n, return the class
    obtained through KNN(n)

    Args:
        data_iterable: iterable of 4 features (not class) to use as input data
        number_of_neighbours: the number of nearest neighbours to check
    """
    iris_df = pd.read_csv(data_path, names=column_names)

    # THIS DOESN'T WORK BECAUSE I ONLY NEED THE FIRST 4 COLUMNS TO OPERATE
    # BUT HAVE TO KEEP THE CLASS IN ORDER TO CLASSIFY. MY DESIRED RESULT WOULD
    # BE A (150, 2) DF, WITH A COLUMN FOR DISTANCE AND THE OTHER THE CLASS OF
    # THE POINT FROM WHICH THE DISTANCE WAS CALCULATED. THIS IS WHERE I'M STUCK
    distances_df = pd.DataFrame()
    distances_df["distance"] = iris_df.apply(
        lambda row: distance_between_vectors(data_iterable, row[:-1]), axis=1, raw=True, result_type="reduce")
    distances_df[column_names[-1]] = iris_df[column_names[-1]]
    distances_df.sort_values(by='distance', inplace=True)
    print("\nResulting df:")
    print(distances_df)

    return Counter(distances_df[column_names[-1]][:number_of_neighbours]).most_common(1)[0][0]


if __name__ == "__main__":
    flower_data, number_of_neighbours = get_data()
    # flower_data = [5.4, 4.4, 1.6, 0.6]  # Iris-setosa
    # flower_data = [7.1, 3.3, 4.6, 1.5]  # Iris-versicolor
    # flower_data = [6.2, 3.4, 6.1, 2.4]  # Iris-virginica

    # number_of_neighbours = 3
    print(
        f"\nClassifying {flower_data} with {number_of_neighbours} neighbours...")
    print(
        f"\nThe data has been classified as: {knn_classify(flower_data, number_of_neighbours)}")
