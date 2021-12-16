from DataParser import DataParser
from HashtagTrainer import HashtagTrainer

data = DataParser("./dual_parsed_data/", vocab_count=300)
translator = data.get_tag_dict()
x, y = data.parse_train()

# Create a HashtagTrainer object with the tweets and their hashtags
b = HashtagTrainer(x, y, translator=translator)

b.minibatch_fit()

test_data, test_labels = data.parse_test()

b.classify_datapoints(test_data, test_labels)