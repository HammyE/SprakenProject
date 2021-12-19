import numpy as np
import pandas
from sklearn.neural_network import MLPClassifier
from DataParser import DataParser


def model_evaluation(results, translator):
    accuracy = np.trace(results) / np.sum(results) * 100
    print(f"Accuracy: {str(accuracy)} %")

    for i, row in enumerate(results):
        if row.sum() == 0:
            precision = "N/A"
        else:
            precision = str(int((row[i] / row.sum()) * 100))

        label = translator[int(i)]

        print("Precision for #" + label + " = " + precision + '%')

    for column in range(4):
        sum = 0
        for i, row in enumerate(results):
            sum += row[column]

        if sum == 0:
            recall = "N/A"
        else:
            recall = str(int((results[column][column] / sum) * 100))

        label = translator[int(column)]

        print("Recall for #" + label + " = " + recall + '%')

    local_labels = []
    for label in range(4):
        local_labels.append(translator[int(label)])
    df = pandas.DataFrame(results, columns=local_labels, index=local_labels, dtype=int)
    print(df)


def main():
    data = DataParser("./parsed_data/", vocab_count=800, tolerance=3)
    translator = data.get_tag_dict()
    x, y = data.parse_train()
    # Create a HashtagTrainer object with the tweets and their hashtags

    nn = MLPClassifier(activation='logistic', hidden_layer_sizes=(10, 15), random_state=1, max_iter=1000)
    nn.fit(x, y)

    test_data, test_labels = data.parse_test()

    predictions = nn.predict(test_data)
    results = np.zeros((4, 4))

    for i in range(len(predictions)):
        pred = int(predictions[i])
        real = int(test_labels[i])
        results[pred, real] += 1

    model_evaluation(results, translator)

    tweet = input("Enter a tweet:")
    # tweet = "exit"
    while tweet != "exit":
        vect = data.get_tweet_vector(f" {tweet} ")
        index = nn.predict(np.transpose(vect.reshape(-1, 1)))
        print()
        print(f"for the tweet: \"{tweet}\" we would use #{translator[int(index)]}")
        print()
        tweet = input("Enter a tweet:")


if __name__ == '__main__':
    main()


