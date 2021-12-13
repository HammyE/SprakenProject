import numpy as np


class DataParser():

    def __init__(self):
        with open(file_name) as file:
            self.__file_raw = file.read()

    def parse_train(self):
        return np.ones([2, 3]), np.ones(2)

    def parse_test(self):
        return np.array(["GenericTweet"]*5), np.array(["GenericHashtag"]*5)
