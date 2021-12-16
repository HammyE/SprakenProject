import csv
import os
import random
import string
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split


class Tweet:

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

    def __init__(self, directory, vocab_count=17):

        self.common_words = ['the', 'to', 'in', 'of', 'a', 'and', 'for', 'you', 'is', 'on', 'this', 'be', 'as', 'should', 'via', 'from', 'that', 'or', 'with', 'will', 'by', 'has', 'have', 'at', 'it']
        self._tweets = []
        self._top_tokens = list()
        self._tags = 0
        self._index_tag_dict = dict()
        self._tag_index_dict = dict()
        self._corpus_occurrences = dict()
        self.vocab_count = vocab_count

        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                self.add_file(filename, directory)

        self._n_features = len(self.get_tweet_vector())
        self._n_values = len(self._tweets)
        self._y = np.array(['']*self._n_values)
        self._X = np.zeros([self._n_values, self._n_features])

        #  print(self._top_tokens)

        for i, tweet in enumerate(self._tweets):
            self._y[i] = self._index_tag_dict[tweet.get_tag()]
            self._X[i, :] = self.get_tweet_vector(tweet.get_text())

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y,
                                                                                    test_size=0.2)  # random_state=42

    def add_file(self, filename, directory):
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
                    if self._corpus_occurrences[token] == 3:
                        self._top_tokens.remove(token)
                local_vocab.pop(token)

    def get_tweet_vector(self, text='hej'):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tests = []
        for token in self._top_tokens:
            tests.append(text.count(' ' + token + ' '))
        # tests.append(len(text))

        return np.array(tests)


    def parse_train(self):
        return self._X_train, self._y_train

    def parse_test(self):
        return self._X_test, self._y_test

    def get_rand_tweet(self):
        return random.sample(self._tweets, 1)[0]

    def get_tag_dict(self):
        return self._tag_index_dict


def main():
    data = DataParser("./parsed_data/", vocab_count=40)
    for i in range(10):
        print(data.get_rand_tweet().get_tag())


if __name__ == '__main__':
    main()
