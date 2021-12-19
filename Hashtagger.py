from DataParser import DataParser
from HashtagTrainer import HashtagTrainer


def main():

    # -------------------------- Träning -------------------------- #
    # Vilka tweets vi läser in och hur många särdrag vi vill använda
    data = DataParser("./parsed_data/", vocab_count=800, tolerance=3)

    # För att kunna översätta från index till hashtag
    translator = data.get_tag_dict()

    # Träningsdatan
    x, y = data.parse_train()

    # Skapar ett HashtagTrainer objekt med de 80% av de nedladdade tweetsen och dess hashtags
    b = HashtagTrainer(x, y, translator=translator)

    # Val av anpassningsmetod
    b.minibatch_fit()

    # -------------------------- Testning -------------------------- #
    # Testdatan
    test_data, test_labels = data.parse_test()

    # Utvärderar modellen på testdatan (20% av de nedladdade tweetsen)
    b.classify_datapoints(test_data, test_labels)

    # -------------------------- Testning - Input -------------------------- #
    tweet = input("Enter a tweet:")
    while tweet != "exit":
        if tweet == "-show":
            b.set_show_vec()
        vect = data.get_tweet_vector(f" {tweet} ")
        index = b.classify_input(vect)
        print()
        print(f"for the tweet: \"{tweet}\" we would use #{translator[int(index)]}")
        print()
        tweet = input("Enter a tweet:")


if __name__ == '__main__':
    main()
