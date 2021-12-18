from DataParser import DataParser
from HashtagTrainer import HashtagTrainer


class HashtagTesting:

    def __init__(self):
        self.trainer = None
        self.data = None

    def train(self):
        """
        Tests the code on a toy example.
        """
        self.data = DataParser("./triple_parsed_data/", vocab_count=30)
        translator = self.data.get_tag_dict()
        x, y = self.data.parse_train()

        # Create a HashtagTrainer object with the tweets and their hashtags
        self.trainer = HashtagTrainer(x, y, translator=translator)

        # Train the model
        self.trainer.minibatch_fit()

    def test(self):
        test_data, test_labels = self.data.parse_test()
        self.trainer.classify_datapoints(test_data, test_labels)

    def predict(self):
        print("------------------------------- Hashtag Predictor --------------(write stop to end)-----------------")
        tweet = input('Next tweet: ')
        while tweet != "stop":
            hashtag = self.trainer.classify_input(self.data.get_tweet_vector(tweet))
            print("Predicted hashtag: #" + hashtag)
            tweet = input('Next tweet: ')


def main():
    testing = HashtagTesting()
    testing.train()
    testing.test()
    testing.predict()


main()
