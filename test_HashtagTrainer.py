from HashtagTrainer import HashtagTrainer


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

main()