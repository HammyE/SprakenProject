from DataParser import DataParser
from HashtagTrainer import HashtagTrainer

def main():
    data = DataParser("./parsed_data/", vocab_count=400)
    translator = data.get_tag_dict()
    x, y = data.parse_train()

    # Create a HashtagTrainer object with the tweets and their hashtags
    b = HashtagTrainer(x, y, translator=translator)

    b.minibatch_fit()

    test_data, test_labels = data.parse_test()

    b.classify_datapoints(test_data, test_labels)

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