import pandas as pd
import numpy as np
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


def ask_for_neighbours():
    """Asks for user input for the number of neighbours
    """
    number_of_neighbours = input(
        "Please indicate, with an integer, the number of neighbours desired\n")
    try:
        number_of_neighbours = int(number_of_neighbours)
    except:
        print("Error during conversion. Please check input value")
        return 0
    return number_of_neighbours


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
    print(iris_df)
    print(data_iterable, number_of_neighbours, iris_df.iloc[0])
    print(distance_between_vectors(data_iterable, iris_df.iloc[0][:-1]))
    # Minimum distance will be sqrt(a**2, b**2, c**2, d**2)
    # where a = f1 - x1, b = f2 - x2... where fn is each feature
    # of the data, in order

    # THIS DOESN'T WORK BECAUSE I ONLY NEED THE FIRST 4 COLUMNS TO OPERATE
    # BUT HAVE TO KEEP THE CLASS IN ORDER TO CLASSIFY. MY DESIRED RESULT WOULD
    # BE A (150, 2) DF, WITH A COLUMN FOR DISTANCE AND THE OTHER THE CLASS OF
    # THE POINT FROM WHICH THE DISTANCE WAS CALCULATED. THIS IS WHERE I'M STUCK
    distances_df = iris_df.apply(lambda row: distance_between_vectors(data_iterable, row[:-1]), axis=1)
    print(distances_df)

    # Now we pick the class from the mode of the first number_of_neighbour
    # elements
    return (distances_df[column_names[-1]][:number_of_neighbours])


if __name__ == "__main__":
    # flower_data = ask_for_flower_data()
    # while len(flower_data) != 4 or not is_float_list(flower_data):
    #     print("*WARNING: Invalid data. Try again*\n")
    #     flower_data = ask_for_flower_data()
    flower_data = [5.2, 4.5, 1.5, 0.5]
    print('\nCorrectly stored data. Proceeding to next step')
    # number_of_neighbours = ask_for_neighbours()
    # while number_of_neighbours <= 0 or type(number_of_neighbours) is not int:
    #     number_of_neighbours = ask_for_neighbours()
    number_of_neighbours = 2
    print(
        f"\nFantastic, we'll proceed to classify with {number_of_neighbours} neighbours. One moment please...")
    print(
        f"The data has been classified as a {knn_classify(flower_data, number_of_neighbours)}")
