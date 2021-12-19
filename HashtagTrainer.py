import math
import random
import numpy as np
import pandas
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


class HashtagTrainer(object):
    """
    Denna klass utför multinomial logistisk regression, den använder sig antingen av
    batch gradient descent, stochastic gradient descent eller minibatch gradient descent
    """

    def __init__(self, x=None, y=None, theta=None, translator=None):
        """
        :Konstruktor: Importerar tweetsen och klasserna för att kunna bygga theta matrisen

        :param x: Input som är en tweets x särdrags matris
        :param y: Klasserna som är en vektor av längd antal datapunkter (tweets)
        :param theta: En redo theta model (istället för x och y)
        """

        #  ------------- Hyperparameters ------------------ #

        self.LEARNING_RATE = 0.1            # Lärningstaktem 0.1
        self.MAX_ITERATIONS = 5000          # Max antal gånger att gå igenom datapunkterna i stochastic gradient descent.
        self.CONVERGENCE_MARGIN = 0.005     # The convergence criterion. 0.005
        self.MINIBATCH_SIZE = 100
        self.lamda = 0.01

        # -------------------------------------------------- #

        if (type(x) is type(None) and type(y) is type(None)) == (type(theta) is type(None)):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif not (type(x) is type(None) or type(y) is type(None)):
            # Antal datapunkter
            self.DATAPOINTS = len(x)

            # Antal särdrag
            self.FEATURES = len(x[0]) + 1

            # Antal klasser
            self.LABELS = len(set(y))

            # Distinkta klasser
            self.labels = list(dict.fromkeys(y))

            # Datapunkterna x Särdragen plus dummy särdragen
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), x), axis=1)

            # Rätt klasser för datapunkterna
            self.y = np.array(y)

            # Vikterna som vi uppdaterar under träningen
            self.theta = np.random.uniform(-1, 1, (self.LABELS, self.FEATURES))

            # Jacoby matrisen
            self.jacoby = np.zeros((self.LABELS, self.FEATURES))

            self.translator = translator

            self.show_vec = False

    def loss(self, x, y):
        """
        Uför loss funktionen givet särdragsvektorerna och klasserna

        :param      x:    Input särdragsvektoerna
        :param      y:    De rätta klasserna
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
        Softmax funktionen
        """
        return np.exp(z) / sum(np.exp(z))

    def compute_gradient_for_all(self):
        """
        Sakapar jacobianen baserat på hela datsettet
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

            # Uppdaterar jacoby matrisen, de riktningarna
            for n in range(0, self.FEATURES):
                self.jacoby[k][n] = k_label_avg[n]

    def compute_gradient(self, datapoint):
        """
        Skapar jacobianen baserat på en enda datapunkt
        """
        for k in range(0, self.LABELS):
            k_label_sum = 0
            hv = (self.softmax(np.dot(self.theta, self.x[datapoint])))[k]
            if self.labels[k] == self.y[datapoint]:
                k_label_sum += self.x[datapoint] * (hv - 1)
            else:
                k_label_sum += self.x[datapoint] * hv

            # Uppdaterar jacoby matrisen, de riktningarna
            for n in range(0, self.FEATURES):
                self.jacoby[k][n] = k_label_sum[n]

    def compute_gradient_minibatch(self, minibatch):
        """
        Skapar jacobianen baserat på en minibatch
        """
        for k in range(0, self.LABELS):
            k_label_sum = 0
            for d in minibatch:
                hv = (self.softmax(np.dot(self.theta, self.x[d])))[k]
                if self.labels[k] == self.y[d]:
                    k_label_sum += self.x[d] * (hv - 1)
                else:
                    k_label_sum += self.x[d] * hv
            k_label_avg = k_label_sum / len(minibatch)

            # Uppdaterar jacoby matrisen, de riktningarna
            for n in range(0, self.FEATURES):
                self.jacoby[k][n] = k_label_avg[n]

    def minibatch_fit(self):
        """
        Utför Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)
        count = 0

        while True:
            minibatch = random.sample(range(len(self.x)), self.MINIBATCH_SIZE)
            self.compute_gradient_minibatch(minibatch)

            # Efter att ha uppdaterat jacobian så stegar vi i motsatt riktning till den brantaste vägen,
            # dvs vi lokaliserar minmumet
            for k in range(1, self.LABELS):
                for n in range(0, self.FEATURES):
                    self.theta[k][n] = self.theta[k][n] - self.LEARNING_RATE * self.jacoby[k][n]

            if count % 500 == 0:
                self.update_plot(self.loss(self.x, self.y))
            count += 1

            # När steget är tillräckligt litet avbryts loopen
            if np.sum(self.jacoby ** 2) < self.CONVERGENCE_MARGIN:
                print(self.theta)
                break

    def stochastic_fit(self):
        """
        Utför Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)
        count = 0

        for i in range(self.MAX_ITERATIONS):
            self.compute_gradient(random.randint(0, self.DATAPOINTS - 1))

            # Efter att ha uppdaterat jacobian så stegar vi i motsatt riktning till den brantaste vägen,
            # dvs vi lokaliserar minmumet
            for k in range(1, self.LABELS):
                for n in range(0, self.FEATURES):
                    self.theta[k][n] = self.theta[k][n] - self.LEARNING_RATE * self.jacoby[k][n]

            if count % 50 == 0:
                self.update_plot(self.loss(self.x, self.y))
            count += 1

    def fit(self):
        """
        Utför Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)
        count = 0

        while True:
            self.compute_gradient_for_all()

            # Efter att ha uppdaterat jacobian så stegar vi i motsatt riktning till den brantaste vägen,
            # dvs vi lokaliserar minmumet
            for k in range(1, self.LABELS):
                for n in range(0, self.FEATURES):
                    self.theta[k][n] = self.theta[k][n] - self.LEARNING_RATE * self.jacoby[k][n]

            if count % 50 == 0:
                self.update_plot(self.loss(self.x, self.y))
            count += 1

            # När steget är tillräckligt litet avbryts loopen
            if np.sum(self.jacoby ** 2) < self.CONVERGENCE_MARGIN:
                break

    def classify_datapoints(self, test_data, test_labels):
        """
        Klassificerar testdatan
        """

        # Lägger på dummy särdraget
        test_data = np.concatenate((np.ones((len(test_labels), 1)), test_data), axis=1)

        if self.FEATURES != len(test_data[0]):
            raise Exception('Felaktig indata')

        # Skapar en matris där diagonalen representerar de korrekta klassficieringarna
        results = np.zeros((self.LABELS, self.LABELS))

        for i in range(len(test_data)):
            probability_vector = self.softmax(np.dot(self.theta, test_data[i]))
            predicted_index = np.argmax(probability_vector, axis=0)
            true_index = self.labels.index(test_labels[i])
            results[predicted_index][true_index] = results[predicted_index][true_index] + 1

        print("------------------------------- The model -------------------------------")
        self.model_evaluation(results)
        if self.translator:
            local_labels = []
            for label in self.labels:
                local_labels.append(self.translator[int(label)])
            df = pandas.DataFrame(results, columns=local_labels, index=local_labels, dtype=int)
        else:
            df = pandas.DataFrame(results, columns=self.labels, index=self.labels, dtype=int)
        print(df)
        print("------------------------------- End of model -------------------------------\n")

    def model_evaluation(self, results):
        """
        Räknar ut modellens accuracy, precision och recall
        """
        # Accuracy
        accuracy = np.trace(results) / np.sum(results) * 100
        print(f"Accuracy: {str(accuracy)} %")

        # Precision
        for i, row in enumerate(results):
            if row.sum() == 0:
                precision = "N/A"
            else:
                precision = str(int((row[i] / row.sum()) * 100))

            if self.translator:
                label = self.translator[int(self.labels[i])]
            else:
                label = self.labels[i]

            print("Precision for #" + label + " = " + precision + '%')

        # Recall
        for column in range(self.LABELS):
            sum = 0
            for i, row in enumerate(results):
                sum += row[column]

            if sum == 0:
                recall = "N/A"
            else:
                recall = str(int((results[column][column] / sum) * 100))

            if self.translator:
                label = self.translator[int(self.labels[column])]
            else:
                label = self.labels[column]

            print("Recall for #" + label + " = " + recall + '%')

    def classify_input(self, vect):
        """
        Klassificerar ny tweets användaren skriver själv
        """
        # Lägger till dummy särdraget
        features = np.concatenate((np.array([1]), vect), axis=0)

        # Fördelningen bland klasserna
        probability_vector = self.softmax(np.dot(self.theta, features))

        # Väljer den mest sannolika klassificeringen
        predicted_index = np.argmax(probability_vector, axis=0)

        # Klassen väljs med hjälp av indexet
        hashtag = self.labels[predicted_index]

        if self.show_vec:
            print(probability_vector)
        return hashtag

    def update_plot(self, *args):
        """
        Tar hand om plottningen
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
        num_axes är antalet variabler som ska plottas
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

    def set_show_vec(self, state=True):
        self.show_vec = state


def main():
    """
    Testar koden med ett litet exempel
    """
    # -------------------------- Träning -------------------------- #
    # Träningsdatan i form av en matris
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
         [0, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 1, 0]]

    x = np.array(x)

    # Och en vektor med klasserna
    y = ["glad", "glad", "glad", "glad", "arg", "arg", "arg", "arg", "ledsen", "ledsen", "ledsen", "ledsen",
         "exalterad", "exalterad", "exalterad", "exalterad"]

    # Skapar ett HashtagTrainer objekt med de 80% av de nedladdade tweetsen och dess hashtags
    b = HashtagTrainer(x, y)

    # Val av anpassningsmetod
    b.fit()

    # -------------------------- Testning -------------------------- #
    # Testdatans särdrag
    test_data = [[1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 1, 1, 1]]

    # Testdatans klasser
    test_labels = ["glad", "exalterad"]

    # Utvärderar modellen
    b.classify_datapoints(test_data, test_labels)


if __name__ == '__main__':
    main()
