import collections

from blinker._utilities import lazy_property


class Dataset:
    def __str__(self):
        return "positive and negative sentimental sentences"

    def __init__(self, positive_dataset_path, negative_dataset_path, vocabulary_size=10000):
        self._positive_dataset_path = positive_dataset_path
        self._negative_dataset_path = negative_dataset_path
        self._vocabulary_size=vocabulary_size
        self._sentences_list = []
        self._tokenized_sentence_list = []
        self._word_dictionary = {}

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

    @lazy_property
    def word_dictionary(self):
        """
        assigns each word a numerical value
        builds a dictionary of words and their assigned numerical value
        :return: tuple of dictionary and word frequency
        """
        words = [w for line in self.labeled_dataset[0] for w in line.split()]

        count = [["RARE", -1], ["BEGIN", -2], ["END", -3]]
        count.extend(collections.Counter(words).most_common(self._vocabulary_size - 1))

        for entry in count:
            self._word_dictionary[entry[0]] = len(self._word_dictionary)
        return self._word_dictionary, count


dataset = Dataset("data/rt-polarity.pos", "data/rt-polarity.neg", 5000)
# for l,d in zip(dataset.labeled_dataset[1][:5],dataset.labeled_dataset[0][:5]):
#     print(l,d)
print(dataset.word_dictionary[1])
print(dataset.word_dictionary[0]["ready"])