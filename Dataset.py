import collections

import copy
from blinker._utilities import lazy_property


class Dataset:
    def __str__(self):
        return "positive and negative sentimental dataset"

    def __init__(self, positive_dataset_path, negative_dataset_path, vocabulary_size=10000):
        self._positive_dataset_path = positive_dataset_path
        self._negative_dataset_path = negative_dataset_path
        self._vocabulary_size = vocabulary_size
        self.sentence_end = "END"
        self.sentence_begin = "BEGIN"
        self.rare_word = "RARE"
        self._sentences_list = []
        self._tokenized_sentence_list = []
        self._word_dictionary = {}
        self._vector_dataset = []

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
        assigns each word[redundant enough to be in the vocabulary] a numerical value
        builds a dictionary of words and their assigned numerical value
        :return: tuple of dictionary and word frequency
        """
        words = [w for line in self.labeled_dataset[0] for w in line.split()]

        count = [[self.rare_word, -1], [self.sentence_begin, -2], [self.sentence_end, -3]]
        count.extend(collections.Counter(words).most_common(self._vocabulary_size - 1))

        for entry in count:
            self._word_dictionary[entry[0]] = len(self._word_dictionary)
        return self._word_dictionary, count

    @lazy_property
    def vector_dataset(self):
        """
        builds vectorized data set and label
        :return: tuple of dataset and label
        """
        vectorized_data = []
        for sentence in self.labeled_dataset[0]:
            vectorized_data.append(self.sentence2vector(sentence))
        return vectorized_data, self.labeled_dataset[1]

    @lazy_property
    def language_model_dataset(self):
        """
        this property holds dataset for language models
        the label of each input is the copy of the input shifted to the the left
        sentence_start will be prepended to the input and sentence_end will be appended to the label
        :return:tuple of inputs and labels
        """
        lmodel_data = copy.deepcopy(self.vector_dataset[0])
        lmodel_label = copy.deepcopy(self.vector_dataset[0])
        for tr_data, lb_data in zip(lmodel_data, lmodel_label):
            tr_data.insert(0,self.word2index(self.sentence_begin))
            lb_data.append(self.word2index(self.sentence_end))
        return lmodel_data, lmodel_label

    def word2index(self, word):
        """
        takes a word and returns the numerical value assigned to the word
        if the number doesn't exist in the word dictionary return 0[RARE WORD]
        :param word:
        :return index: assigned numerical value
        """
        if word in self.word_dictionary[0]:
            return self.word_dictionary[0][word]
        else:
            return self.word_dictionary[0][self.rare_word]

    def sentence2vector(self, sentence_list):
        """
        each word in the list will be converted to its numerical value
        :param sentence_list: a list of words[sentence]
        :return: a list of numbers
        """
        vector = []
        for word in sentence_list.split(" "):
            vector.append(self.word2index(word))
        return vector


dataset = Dataset("data/rt-polarity.pos", "data/rt-polarity.neg", 10000)

for l,d in zip(dataset.labeled_dataset[1][:5],dataset.labeled_dataset[0][:5]):
    print(l,d)
print("\n\n")

print(dataset.word_dictionary[1])
for l,d in zip(dataset.vector_dataset[1][:5],dataset.vector_dataset[0][:5]):
    print(l,d)
print("\n\n")

for d,l in zip(dataset.language_model_dataset[0][:5],dataset.language_model_dataset[1][:5]):
    print(d)
    print(l)
    print("------------")