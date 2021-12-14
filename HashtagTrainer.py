import math
import random
import numpy as np
import pandas
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


class HashtagTrainer(object):
    """
    This class performs multinomial logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    def __init__(self, x=None, y=None, theta=None):
        """
        :Constructor: Imports the data and labels needed to build theta.

        :param x: The input as a DATAPOINT*FEATURES array.
        :param y: The labels as a DATAPOINT array.
        :param theta: A ready-made model. (instead of x and y)
        """

        #  ------------- Hyperparameters ------------------ #

        self.LEARNING_RATE = 0.1  # The learning rate. 0.01
        self.CONVERGENCE_MARGIN = 0.01  # The convergence criterion. 0.001
        self.MAX_ITERATIONS = 1000  # Maximal number of passes through the datapoints in stochastic gradient descent.
        self.MINIBATCH_SIZE = 1000  # Minibatch size (only for minibatch gradient descent)

        # -------------------------------------------------- #

        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Number of labels.
            self.LABELS = len(set(y))

            # Distinct labels.
            self.labels = list(dict.fromkeys(y))

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            # self.theta = np.random.uniform(-1, 1, (self.LABELS, self.FEATURES))
            self.theta = np.ones((self.LABELS, self.FEATURES))

            # The current jacoby matrix.
            self.jacoby = np.zeros((self.LABELS, self.FEATURES))

    def loss(self, x, y):
        """
        Computes the loss function given the input features x and labels y

        :param      x:    The input features
        :param      y:    The correct labels
        """
        the_sum = 0

        for k in range(0, self.LABELS):
            for d in range(0, self.DATAPOINTS):
                if self.labels[k] == y[d]:
                    the_sum += -math.log((self.softmax(np.dot(self.theta, x[d])))[k])

        final_sum = the_sum / self.DATAPOINTS
        return final_sum

    def softmax(self, z):
        """
        The softmax function.
        """
        return np.exp(z) / sum(np.exp(z))

    def compute_jacobian(self):
        """
        Computes the jacobian matrix based on the entire dataset
        (used for batch gradient descent).
        """
        for k in range(0, self.LABELS):
            k_label_sum = 0
            for d in range(0, self.DATAPOINTS):
                hv = (self.softmax(np.dot(self.theta, self.x[d])))[k]
                if self.labels[k] == self.y[d]:
                    k_label_sum += self.x[d] * (hv - 1)
                else:
                    k_label_sum += self.x[d] * hv
            k_label_avg = k_label_sum / self.DATAPOINTS
            for n in range(0, self.FEATURES):
                self.jacoby[k][n] = k_label_avg[n]

    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)
        while True:
            self.compute_jacobian()

            # Time to update the weights, we'll go in opposite direction of the steepest route
            # meaning we're locating the minimum
            for k in range(1, self.LABELS):
                for n in range(0, self.FEATURES):
                    self.theta[k][n] = self.theta[k][n] - self.LEARNING_RATE * self.jacoby[k][n]

            self.update_plot(self.loss(self.x, self.y))

            # When the step is small enough it means we're almost at the minimum
            if np.sum(self.jacoby ** 2) < self.CONVERGENCE_MARGIN:
                break

    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        if self.FEATURES != len(test_data[0]):
            print("Error: Felaktig indata")

        results = np.zeros((self.LABELS, self.LABELS))

        for i in range(len(test_data)):
            probability_vector = self.softmax(np.dot(self.theta, test_data[i]))
            predicted_index = np.argmax(probability_vector, axis=0)
            true_index = self.labels.index(test_labels[i])
            results[predicted_index][true_index] = results[predicted_index][true_index] + 1
            print(probability_vector)

        # Andel av gissningarna som blir rätt
        correct_ones = 0
        for i in range(self.LABELS):
            correct_ones += results[i][i]
        accuracy = (correct_ones / np.sum(results)) * 100
        print("Accuracy: " + str(accuracy) + "%")

        row_labels = [self.labels[0], self.labels[1], self.labels[2], self.labels[3]]
        column_labels = [self.labels[0], self.labels[1], self.labels[2], self.labels[3]]
        df = pandas.DataFrame(results, columns=column_labels, index=row_labels, dtype=int)
        print(df)

    def update_plot(self, *args):
        """
        Handles the plotting
        """

        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)

    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines = []

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5,
                                            markersize=4)


def main():
    """
    Tests the code on a toy example.
    """
    # --------------------- Training ---------------------- #
    # The training-tweets coded in to their features
    x = [[1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 2, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1, 0]]

    # The training-tweets' hashtags/labels/classes
    y = ["glad", "glad", "glad", "glad", "arg", "arg", "arg", "arg", "ledsen", "ledsen", "ledsen", "ledsen",
         "exalterad", "exalterad", "exalterad", "exalterad"]

    # Create a HashtagTrainer object with the tweets and their hashtags
    b = HashtagTrainer(x, y)

    # Train the model
    b.fit()

    # --------------------- Testing ---------------------- #
    # These test-tweets:
    # "vi alla vill gå på denna fest och hoppas på att få le och skratta #glad"
    # "jag är så taggad och uppspelt att jag tror jag kommer böla #exalterad"
    # Has been coded into their features
    test_data = [[1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 1, 1, 1]]

    # The test-tweet's hashtags
    test_labels = ["glad", "exalterad"]

    b.classify_datapoints(test_data, test_labels)


if __name__ == '__main__':
    main()
