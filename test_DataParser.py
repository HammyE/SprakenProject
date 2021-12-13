from unittest import TestCase

from DataParser import DataParser


class TestDataParser(TestCase):
    def test_parse_train(self):
        parser = DataParser("testDataset.txt")
        features, labels = parser.parse_train()
        correct_features = [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1],
                            [1, 1, 0],
                            [1, 0, 1]]
        correct_labels = ["Trump2020", "Biden2020", "Trump2020", "Biden2020", "Biden2020"]
        TestCase.assertEqual(self, features, correct_features)
        TestCase.assertEqual(self, labels, correct_labels)

    def test_parse_test(self):
        parser = DataParser("testDataset.txt")
        tweets, hashtags = parser.parse_test()
        correct_tweets = ["Donald Trump is my best friend #Trump2020",
                          "Joe Biden is so hot lol #Biden2020",
                          "Why can't libtards just understand that we want to MAGA #Trump2020",
                          "Trans rights are human rights! #Biden2020",
                          "I love how a president can rape women and get away with it #Biden2020"]
        correct_hashtags = ["Trump2020", "Biden2020", "Trump2020", "Biden2020", "Biden2020"]
        TestCase.assertEqual(self, tweets, correct_tweets)
        TestCase.assertEqual(self, hashtags, correct_hashtags)
