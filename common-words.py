#!/usr/bin/env python3
# encoding: utf-8

import argparse
import sys
from operator import itemgetter

import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from tabulate import tabulate

STOPWORDS = set(stopwords.words('english'))

SORT_ALPHA = 'alpha'
SORT_OCCURRENCES = 'occurrences'
SORT_LENGTH = 'length'

DEFAULT_STOPWORDS = False
DEFAULT_SORT = SORT_OCCURRENCES


def main():
    parser = argparse.ArgumentParser(description='Compare texts to find common words and their frequencies')

    parser.add_argument('files', nargs='+', type=argparse.FileType('rU'),
                        help='text files to compare')

    parser.add_argument('--include-stopwords', action='store_true', default=DEFAULT_STOPWORDS,
                        help='include very common words in results (default: {})'
                        .format(DEFAULT_STOPWORDS))

    parser.add_argument('--word-frequency-list', type=argparse.FileType('rU'),
                        help='''providing a text file containing "word;frequency"
                        lines will print that frequency data next to the word
                        in output, and sort on it''')

    parser.add_argument('--sort', type=str, default=False,
                        choices=[SORT_ALPHA, SORT_OCCURRENCES, SORT_LENGTH],
                        help='''how to sort output; by word alphabetically,
                        number of occurrences, or by word length (default: {})'''
                        .format(DEFAULT_SORT))

    args = parser.parse_args()

    if len(args.files) < 2:
        sys.exit('You must specify at least two text files to compare')

    # Instead of passing the default sort order as the `default` kwarg to the
    # parser argument, we do it this way so that we can later determine if a
    # sort order was passed explicitly by the user or not.
    if args.sort:
        sort_order = args.sort
    else:
        sort_order = DEFAULT_SORT

    # Download required nltk data if needed.
    nltk.download('punkt')
    if not args.include_stopwords:
        nltk.download('stopwords')

    # Create the tokenized lists of words from all texts.
    texts = [f.read() for f in args.files]
    texts = [t.lower() for t in texts]
    tokens = [nltk.word_tokenize(t) for t in texts]

    # Only keep words consisting entirely of letters (i.e. remove punctuation,
    # numbers, etc.)
    tokens = [[word for word in t if word.isalpha()] for t in tokens]

    # Save number of occurrences for each word in each text for later reference,
    # which could as well be through a collections.Counter but FreqDist will
    # cover more cases if expanded on later.
    fdists = [FreqDist(t) for t in tokens]

    # Build list of matches, i.e. tokens occuring in all texts/lists of tokens.
    tokens = [set(t) for t in tokens]
    matches = {word for word in set.intersection(*tokens)}

    # Remove stopwords, i.e. very common ones.
    if not args.include_stopwords:
        matches -= STOPWORDS

    print('Found {} common words'.format(len(matches)))

    # Make a list of (word, occurrences, length) tuples for final output. `word`
    # is the word itself, `occurrences` the number of times it occurs in the
    # source text, and `length` is the length of the word.
    matches = [(word, min([f.get(word) for f in fdists]), len(word)) for word in matches]
    headers = ['word', 'occurrences', 'length']

    if args.word_frequency_list:
        # Read each word;frequency line as key and value for fast lookup.
        frequencies = {}
        for line in args.word_frequency_list:
            term, frequency = line.split(';')
            frequencies[term] = float(frequency)

        # Go through the list of matches and look up each word's frequency, and
        # build a new list of matches with this value included.
        matches = [(word, occurrences, length, frequencies.get(word, 'unknown'))
                   for (word, occurrences, length) in matches]
        headers.append('in literature')

    # Sort output accordingly.
    if args.word_frequency_list and not args.sort:
        matches = sorted(matches, key=itemgetter(3, 0))
    elif sort_order == SORT_ALPHA:
        matches = sorted(matches, key=itemgetter(0))
    elif sort_order == SORT_OCCURRENCES:
        matches = sorted(matches, key=itemgetter(1, 0), reverse=True)
    elif sort_order == SORT_LENGTH:
        matches = sorted(matches, key=itemgetter(2, 0), reverse=True)

    # Finally, print results in a nice table.
    print(tabulate(matches, headers=headers))

if __name__ == '__main__':
    main()
