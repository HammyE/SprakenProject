import csv
import os
import random
import string
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split


class Tweet:
    """A class for the storage of tweets"""

    def __init__(self, text, hashtag):
        self._text = text
        self._hashtag = hashtag

    def __str__(self):
        return self._text + ' #' + str(self._hashtag)

    def set_text(self, text):
        self._text = text

    def add_hashtag(self, tag):
        self._hashtag = tag

    def get_tag(self):
        return self._hashtag

    def get_text(self):
        return self._text


class DataParser:
    """This class parses a set of corpuses to create a feature vectors for the hashtagger."""

    def __init__(self, directory, vocab_count=100, tolerance=3):
        """Returns a DataParser object tailored to the directory entered.

        :param directory:    Stores .txt files with tweets grouped by hashtag. The format for the files should
                            be tag.txt and contain the tweets without the tag, in lower case, separated by row.

        :param vocab_count:  The amount of keywords to be considered per hashtag

        :param tolerance:    The amount of times a keyword is allowed to appear as a top word in a corpus. """

        self.common_words = ['the', 'to', 'in', 'of', 'a', 'and', 'for', 'you', 'is', 'on', 'this', 'be', 'as', 'should', 'via', 'from', 'that', 'or', 'with', 'will', 'by', 'has', 'have', 'at', 'it']
        self._tweets = []
        self._top_tokens = list()
        self._tags = 0
        self._index_tag_dict = dict()
        self._tag_index_dict = dict()
        self._corpus_occurrences = dict()
        self.vocab_count = vocab_count
        self._tolerance = tolerance

        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                self.add_file(filename, directory)

        self._n_features = len(self.get_tweet_vector())
        self._n_values = len(self._tweets)
        self._y = np.array([0]*self._n_values)
        self._X = np.zeros([self._n_values, self._n_features])

        #  print(self._top_tokens)

        print(self._top_tokens)

        for i, tweet in enumerate(self._tweets):
            self._y[i] = self._index_tag_dict[tweet.get_tag()]
            self._X[i, :] = self.get_tweet_vector(tweet.get_text())

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y,
                                                                                    test_size=0.2)  # random_state=42

    def add_file(self, filename, directory):
        '''Parses an individual file.'''
        tag = filename.replace(".txt", '')
        self._index_tag_dict[tag] = self._tags
        self._tag_index_dict[self._tags] = tag
        self._tags += 1
        with open(os.path.join(directory, filename)) as file:
            local_vocab = defaultdict(int)
            for raw_tweet in file.read().split("\n"):
                raw_tweet = ''.join(filter(str.isascii, raw_tweet))
                raw_tweet = raw_tweet.translate(str.maketrans('', '', string.punctuation))
                self._tweets.append(Tweet(raw_tweet, tag))
                for token in raw_tweet.split(' '):
                    if len(token) > 0 and token not in self.common_words:
                        local_vocab[token] += 1
            for i in range(self.vocab_count):
                token = max(local_vocab, key=local_vocab.get)
                if token not in self._top_tokens:
                    self._corpus_occurrences[token] = 1
                    self._top_tokens.append(token)
                else:
                    self._corpus_occurrences[token] += 1
                    if self._corpus_occurrences[token] == self._tolerance:
                        self._top_tokens.remove(token)
                local_vocab.pop(token)

    def get_tweet_vector(self, text='hej'):
        """Creates a feature vector for the string argument

        :param String text: The string from which the vector is created.
        :returns an numpy array of the features"""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tests = []
        for token in self._top_tokens:
            tests.append(text.count(' ' + token + ' '))

        return np.array(tests)


    def parse_train(self):
        """:returns: The training features and labels"""
        return self._X_train, self._y_train

    def parse_test(self):
        """:returns: The test features and labels"""
        return self._X_test, self._y_test

    def get_rand_tweet(self):
        """:returns: A random tweet from the corpus"""
        return random.sample(self._tweets, 1)[0]

    def get_tag_dict(self):
        """:returns: A dictionary to decode the numerical label to the original tag."""
        return self._tag_index_dict


def main():
    data = DataParser("./parsed_data/", vocab_count=40)
    for i in range(10):
        print(data.get_rand_tweet().get_tag())


if __name__ == '__main__':
    main()
