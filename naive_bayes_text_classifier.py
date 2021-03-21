import nltk
import numpy as np
import operator
import random
import itertools

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

from nltk.tokenize import RegexpTokenizer

TRAIN_SPLIT = .8

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class NBTC:
    def __init__(self, params):
        self.word_counts = {}
        self.corpus_dist = {}
        self.logprior = {}
        self.loglikelihood = {}
        self.vocab = set()
        self.params = params

    def get_doc_vocab_freq(self, doc_string):

        words = self.process_words(doc_string)
        doc_vocab = {}
        for word in words:
            if word in doc_vocab:
                doc_vocab[word] = doc_vocab[word] + 1
            else:
                doc_vocab[word] = 1
        return doc_vocab
    def process_words(self, doc_string):

        if self.params["lower-case"]:
            doc_string = doc_string.lower()

        if self.params["tokenizer"] == "nltk":
            words = nltk.word_tokenize(doc_string)
        else:
            tokenizer = RegexpTokenizer(r'\w+')
            words = tokenizer.tokenize(doc_string)

        if self.params["stopwords"]:
            stopword = stopwords.words("english")
            words = [word for word in words if word not in stopword]

        if self.params["lemmatizer"]:
            wordnet_lemmatizer = WordNetLemmatizer()
            words = [wordnet_lemmatizer.lemmatize(word) for word in words]

        if self.params["stemmer"]:
            snowball_stemmer = SnowballStemmer("english")
            words = [snowball_stemmer.stem(word) for word in words]

        return words

    def process_document(self, doc_path, doc_label):
        document = open(doc_path)
        doc_vocab = self.get_doc_vocab_freq(document.read())
        if doc_label in self.word_counts.keys():
            for key, value in doc_vocab.items():
                if key in self.word_counts[doc_label]:
                    self.word_counts[doc_label][key] += value
                else:
                    self.word_counts[doc_label][key] = value
        else:
            self.word_counts[doc_label] = doc_vocab

    def calc_loglikelihood(self):

        for category_name, category in self.word_counts.items():
            word_total = 0
            for word, count in category.items():
                word_total += count

            denom = word_total + len(self.vocab)*self.params["smoothing"]
            loglikelihoods = {}
            for word in self.vocab:
                if word in category.keys():
                    loglikelihoods[word] = np.log((category[word] + self.params["smoothing"])/denom)
                else:
                    loglikelihoods[word] = np.log(self.params["smoothing"]/denom)

            self.loglikelihood[category_name] = loglikelihoods

    def calc_logprior(self, total):
        for category, count in self.corpus_dist.items():
            self.logprior[category] = np.log(count/total)

    def calc_vocab(self):
        for category, words in self.word_counts.items():
            for word, counts in words.items():
                self.vocab.add(word)

    def train(self, docs):

        self.corpus_dist = {}
        total_docs = 0
        for doc_info in docs:
            total_docs += 1
            temp = doc_info.split(" ")

            doc_path = temp[0]
            doc_label = temp[1][:-1]

            if doc_label in self.corpus_dist:
                self.corpus_dist[doc_label] = self.corpus_dist[doc_label] + 1
            else:
                self.corpus_dist[doc_label] = 1
            self.process_document(doc_path, doc_label)

        self.calc_vocab()
        self.calc_loglikelihood()
        self.calc_logprior(total_docs)



    def predict(self, doc_info):
        temp = doc_info.split(" ")
        doc_path = temp[0]
        if doc_path[-1] == "\n":
            doc_path = doc_path[:-1]
        #doc_label = temp[1][:-1]

        f = open(doc_path)
        doc_words = self.process_words(f.read())

        summ = {}
        for category in self.logprior.keys():
            # Try removing prior prob
            summ[category] = 0
            #summ[category] = self.logprior[category]
            for word in doc_words:
                if word in self.loglikelihood[category]:
                    summ[category] += self.loglikelihood[category][word]
        predicted_label = max(summ.items(), key=operator.itemgetter(1))[0]

        return predicted_label

    def test(self, test_docs):
        output_file = input("What would you like the output file to be called?")
        output = open(output_file, "w")

        if len(test_docs[1].split(" ")) > 1:
            with_labels = True
        else:
            with_labels = False

        num_correct = 0
        num_docs = 0

        for doc_info in test_docs:

            if with_labels:
                num_docs += 1
                num_correct += 1 if self.predict(doc_info) == doc_info.split(" ")[1][:-1] else 0
                print("Accuracy: " + str(num_correct/num_docs))

            else:
                output.write(doc_info[:-1] + " " + self.predict(doc_info) + "\n")

        return num_correct / num_docs if num_docs > 0 else 0



def get_input_files():
    one_file = input("Will it be one file for training and testing? (y/n)")
    if one_file == "y":
        doc_file_path = input("What is the file path? ")
        doc_file = open(doc_file_path, "r")
        all_docs = doc_file.readlines()
        random.shuffle(all_docs)

        train_docs = all_docs[:int(len(all_docs) * TRAIN_SPLIT)]
        test_docs = all_docs[int(len(all_docs) * TRAIN_SPLIT):]
    else:
        train_file_path = input("What is the train file path? ")
        train_file = open(train_file_path, "r")
        train_docs = train_file.readlines()

        test_file_path = input("What is the test file path? ")
        test_file = open(test_file_path, "r")
        test_docs = test_file.readlines()


    return (train_docs, test_docs)


def sweep_params():
    one_file = True
    train_file_path = "./TC_provided/corpus1_train.labels"
    test_file_path = "./TC_provided/corpus1_test.labels"

    params = {
        "tokenizer": ["only-letters"],
        "stopwords": [True],
        "stemmer": [False],
        "lemmatizer": [False],
        "smoothing": [.0001, .001, .01, .1, .25, .15, .3, .4],
        "lower-case": [True]
    }

    keys, values = zip(*params.items())
    param_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    max_average_accuracy = 0
    best_params = {}

    for param_permutation in param_permutations:
        print(param_permutation)
        accuracies = []
        for i in range(10):
            if one_file:
                train_file = open(train_file_path)
                all_docs = train_file.readlines()
                random.shuffle(all_docs)

                train_docs = all_docs[:int(len(all_docs) * TRAIN_SPLIT)]
                test_docs = all_docs[int(len(all_docs) * TRAIN_SPLIT):]

                nbtc = NBTC(param_permutation)
                nbtc.train(train_docs)
                accuracies.append(nbtc.test(test_docs))
            else:
                train_file = open(train_file_path, "r")
                train_docs = train_file.readlines()
                nbtc = NBTC(param_permutation)
                nbtc.train(train_docs)

                test_file = open(test_file_path)
                test_docs = test_file.readlines()
                accuracies.append(nbtc.test(test_docs))
        avg_accuracy = sum(accuracies) / len(accuracies)
        print("Average accuracy: " + str(avg_accuracy))

        if (avg_accuracy > max_average_accuracy):
            max_average_accuracy = avg_accuracy
            best_params = param_permutation

    print("Max average accuracy: " + str(max_average_accuracy))
    print("Best params: " + str(best_params))

if __name__ == "__main__":
    nbtc = NBTC({
        "tokenizer": "only-letters",
        "stopwords": True,
        "stemmer": False,
        "lemmatizer": True,
        "smoothing": .25,
        "lower-case": True
    })

    train_docs, test_docs = get_input_files()

    nbtc.train(train_docs)
    nbtc.test(test_docs)

    # params = {
    #     "tokenizer": ["nltk", "only-letters"],
    #     "stopwords": [True, False],
    #     "stemmer": [True, False],
    #     "lemmatizer": [True, False],
    #     "smoothing": [.1 ,.25, .5, 1, 2, 5, 10],
    #     "lower-case": [True, False]
    # }
