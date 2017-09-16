
import csv
import os
import sys
from collections import namedtuple, Counter
import random

import toolz
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy

Example = namedtuple('Example', ('category', 'text', 'sentiment'))

AIRLINE_DATA_SOURCE = os.environ.get('AIRLINE_DATA_SOURCE', './data/airline-tweets.csv')
COMBINED_DATA_SOURCE = os.environ.get('COMBINED_DATA_SOURCE', './data/combined-sentiment-dataset.csv')
GOP_DATA_SOURCE = os.environ.get('GOP_DATA_SOURCE', './data/gop-debate-sentiment.csv')

DATASET_PATH = 'data/text_data_{}.csv'

HELDOUT_PERCNET = 0.1
TEST_PERCENT = 0.2


def read_from_file(fpath, row_to_example):
    with open(fpath) as infile:
        reader = csv.reader(infile)
        _headers = next(reader)
        for row in reader:
            yield row_to_example(row)


def airline_row_to_example(row):
    category = '__twitter_airlines_' + row[3].replace(' ', '_').lower()
    text = row[10]
    # TODO: do we care about sentiment confidence?
    sentiment = row[1].lower()
    return Example(category, text, sentiment)


def gop_row_to_example(row):
    category = '__twitter_gop_debates_' + row[1].replace(' ', '_').lower()
    text = row[15]
    sentiment = row[5].lower()
    return Example(category, text, sentiment)


def combined_row_to_example(row):
    category = '__combined_data'
    text = row[3]
    sentiment = 'positive' if int(row[0]) == 1 else 'negative'
    return Example(category, text, sentiment)


def prep_data():
    print("Reading data...")
    all_examples = list(tqdm(toolz.concat([
        read_from_file(AIRLINE_DATA_SOURCE, airline_row_to_example),
        read_from_file(GOP_DATA_SOURCE, gop_row_to_example),
        read_from_file(COMBINED_DATA_SOURCE, combined_row_to_example),
    ])))
    random.shuffle(all_examples)

    print("Partitioning data...")
    train_data = []
    test_data = []
    heldout_data = []

    for example in tqdm(all_examples):
        roll = random.random()
        if roll < HELDOUT_PERCNET:
            heldout_data.append(example)
        elif roll < HELDOUT_PERCNET + TEST_PERCENT:
            test_data.append(example)
        else:
            train_data.append(example)

    for kind, data in [('heldout', heldout_data), ('train_test', test_data + train_data)]:
        outpath = DATASET_PATH.format(kind)
        print("Writing {} data to {}...".format(kind, outpath))
        with open(outpath, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['category', 'text', 'is_negative', 'is_neutral', 'is_positive'])
            for row in tqdm(data):
                writer.writerow(row)

    print("Done!")


def read_train_test_dataset():
    with open(DATASET_PATH.format('train_test')) as infile:
        reader = csv.reader(infile)
        _headers = next(reader)
        return [Example(*row) for row in tqdm(reader)]


def train_mnb():
    """ Trains and evaluates a multinomial naive bayes model """
    print("Reading dataset...")
    data = read_train_test_dataset()
    random.shuffle(data)

    texts = ['{} {}'.format(example.category, example.text).strip().lower() for example in data]
    sentiments = [example.sentiment for example in data]

    print("Sentiment distributions:", Counter(sentiments))

    # Fit source text and sentiment encoders, transform data
    print("Fitting text and label encoders...")
    text_encoder = CountVectorizer(max_features=2**12)  # 2**15 = 32768
    label_encoder = LabelEncoder()

    x = text_encoder.fit_transform(texts)
    y = label_encoder.fit_transform(sentiments)

    kfold = StratifiedKFold(n_splits=5)
    class_cats = ['{} {}'.format(example.category, example.sentiment) for example in data]

    print("Training and evaluating MNB models...")
    fold_num = 1
    last_model = None
    for train_idx, test_idx in kfold.split(x, class_cats):
        print("Epoch {}".format(fold_num))
        train_x, test_x = x[train_idx], x[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]

        # Water down combined dataset due to poor quality
        combined_idx = text_encoder.transform(['__combined_data']).argmax()
        train_weights = numpy.array(1 - 0.98 * train_x[:, combined_idx].todense())[:, 0]
        test_weights = numpy.array(1 - 0.98 * test_x[:, combined_idx].todense())[:, 0]
        print("train weights", Counter(train_weights))

        model = MultinomialNB()
        model.fit(train_x, train_y, sample_weight=train_weights)
        print("Model score:")
        print(model.score(test_x, test_y, sample_weight=test_weights))
        print(Counter(model.predict(test_x)))
        last_model = model
        fold_num += 1

    print("Input some text to have it's sentiment predicted...")
    try:
        while True:
            text = input("> ").strip()
            encoded_text = text_encoder.transform([text])
            print("Sentiment: {}".format(label_encoder.inverse_transform(last_model.predict(encoded_text))))
    except KeyboardInterrupt:
        pass
    print("Done!")



def main(mode):
    """ Combine data, learn vocab, and  """ 
    if mode == 'prep':
        prep_data()
    if mode == 'train-mnb':
        train_mnb()
    else:
        raise NotImplementedError('Mode "{}"" not implemented yet.'.format(mode))


if __name__ == '__main__':
    main(sys.argv[1])
