from blinker._utilities import lazy_property


class Dataset:
    def __str__(self):
        return "positive and negative sentimental sentences"

    def __init__(self, positive_dataset_path, negative_dataset_path):
        self._positive_dataset_path = positive_dataset_path;
        self._negative_dataset_path = negative_dataset_path;
        self._sentences_list = []
        self._tokenized_sentence_list = []

    @lazy_property
    def labeled_dataset(self):
        """
        reads positive and negative sentimental files.
        and constructs labels for each sentence 1for positive 0 for negative
        :first item in the labels is the sentiment of first item in dataset
        :return: tuple of dataset_list containing sentences and label_list containing sentiment
        """
        pos_data = []
        neg_data = []

        with open(self._positive_dataset_path, 'r', encoding="latin-1") as pos_file:
            for line in pos_file:
                pos_data.append(line)

        with open(self._negative_dataset_path, 'r', encoding="latin-1") as neg_file:
            for line in neg_file:
                neg_data.append(line)

        tot_data = pos_data + neg_data
        labels = [1] * len(pos_data) + [0] * len(neg_data)

        print(len(labels), len(tot_data))

        return tot_data, labels

dataset = Dataset("data/rt-polarity.pos", "data/rt-polarity.neg")
for l,d in zip(dataset.labeled_dataset[1][:5],dataset.labeled_dataset[0][:5]):
    print(l,d)