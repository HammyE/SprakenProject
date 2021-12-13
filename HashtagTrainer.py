import numpy as np

class HashtagTrainer:
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
        self.CONVERGENCE_MARGIN = 0.001  # The convergence criterion. 0.001
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

            # Set of labels.
            self.labels = list(set(y))

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, (self.LABELS, self.FEATURES))

            # The current gradient.
            self.jacoby = np.zeros((self.LABELS, self.FEATURES))

    def softmax(self, z):
        """
        The softmax function.
        """
        return np.exp(z) / sum(np.exp(z))

    def conditional_prob(self, label, datapoint):
        # """
        # Computes the conditional probability P(label|datapoint)
        # """
        # temp = self.softmax(np.dot(self.theta, self.x[datapoint]))
        # return temp[label]

    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """
        for k in range(0, self.LABELS):
            v_1 = 0
            for d in range(0, self.DATAPOINTS):
                v_2 = (self.softmax(np.dot(self.theta, self.x[d])))[k]
                if self.labels[k] == self.y[d]:
                    v_1 += self.x[d]*(v_2-1)
                else:
                    v_1 += self.x[d]*v_2
            v_1 = v_1 / self.DATAPOINTS
            for n in range(0, self.FEATURES):
                self.jacoby[k][n] = v_1[n]

    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        while True:
            self.compute_gradient_for_all()

            for k in range(1, self.LABELS):
                for n in range(0, self.FEATURES):
                    self.theta[k][n] = self.theta[k][n] - self.LEARNING_RATE * self.jacoby[k][n]

            if np.sum(self.jacoby ** 2) < self.CONVERGENCE_MARGIN:
                break

    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        # print('Model parameters:')
        #
        # print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))
        #
        # self.DATAPOINTS = len(test_data)
        #
        # self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        # self.y = np.array(test_labels)
        # confusion = np.zeros((self.FEATURES, self.FEATURES))
        #
        # for d in range(self.DATAPOINTS):
        #     prob = self.conditional_prob(1, d)
        #     predicted = 1 if prob > .5 else 0
        #     confusion[predicted][self.y[d]] += 1
        #
        # print('                       Real class')
        # print('                 ', end='')
        # print(' '.join('{:>8d}'.format(i) for i in range(2)))
        # for i in range(2):
        #     if i == 0:
        #         print('Predicted class: {:2d} '.format(i), end='')
        #     else:
        #         print('                 {:2d} '.format(i), end='')
        #     print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))

    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.jacoby]))


def main():
    """
    Tests the code on a toy example.
    """
    x = [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 1],
         [1, 1, 0],
         [1, 0, 1]]

    #  Encoding of the correct classes for the training material
    y = ["Trump2020", "Biden2020", "Trump2020", "Biden2020", "Biden2020"]
    b = HashtagTrainer(x, y)
    b.fit()
    b.classify_datapoints(x, y)
    # b.print_result()

main()
