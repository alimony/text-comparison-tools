#!/usr/bin/env python3
# encoding: utf-8

import argparse
import itertools
import sys
from operator import itemgetter

import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize.punkt import PunktSentenceTokenizer
from tabulate import tabulate

STOPWORDS = set(stopwords.words('english'))

SORT_ALPHA = 'alpha'
SORT_FREQ = 'frequency'
SORT_OCCURRENCES = 'occurrences'
SORT_LENGTH = 'length'

DEFAULT_PUNCTUATION = '()' + ''.join(PunktSentenceTokenizer.PUNCTUATION)
DEFAULT_STOPWORDS = False
DEFAULT_SORT = SORT_OCCURRENCES


def main():
    parser = argparse.ArgumentParser(description='Compare texts to find common words and their frequencies')

    parser.add_argument('files', nargs='+', type=argparse.FileType('rU'),
                        help='text files to compare')

    parser.add_argument('--include-stopwords', action='store_true', default=DEFAULT_STOPWORDS,
                        help='include very common words in results (default: {})'
                        .format(DEFAULT_STOPWORDS))

    parser.add_argument('--sort', type=str, default=DEFAULT_SORT,
                        choices=[SORT_ALPHA, SORT_FREQ, SORT_OCCURRENCES, SORT_LENGTH],
                        help='''how to sort output; by word alphabetically, by its
                        frequency in the given texts, number of occurrences,
                        or by word length (default: {})'''
                        .format(DEFAULT_SORT))

    args = parser.parse_args()

    if len(args.files) < 2:
        sys.exit('You must specify at least two text files to compare')

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

    # Save a list of all occuring tokens, including duplicates, so we can
    # calculate the frequency of a token in the original texts, as well as just
    # the number of occurrences since FreqDist inherits collections.Counter.
    all_tokens = itertools.chain.from_iterable(tokens)
    fdist = FreqDist(all_tokens)

    # Build list of matches, i.e. tokens occuring in all texts/lists of tokens.
    tokens = [set(t) for t in tokens]
    matches = {word for word in set.intersection(*tokens)}

    # Remove stopwords, i.e. very common ones.
    if not args.include_stopwords:
        matches -= STOPWORDS

    print('Found {} common words'.format(len(matches)))

    # TODO: nltk.metrics.scores.log_likelihood(reference, test)[source]
    # Add parameter to supply a custom frequency list, e.g. one from 1902.

    # Make a list of (word, frequency, occurrences, length) tuples for final
    # output. `word` is the word itself, `frequency` how often it occurs in the
    # source text, `occurrences` in absolute numbers, and `length` is the length
    # of the word.
    matches = [(word, round(fdist.freq(word) * 100, 5), fdist.get(word), len(word)) for word in matches]

    # Sort output accordingly.
    if args.sort == SORT_ALPHA:
        matches = sorted(matches, key=itemgetter(0))
    elif args.sort == SORT_FREQ:
        matches = sorted(matches, key=itemgetter(1, 0))
    elif args.sort == SORT_OCCURRENCES:
        matches = sorted(matches, key=itemgetter(2, 0))
    elif args.sort == SORT_LENGTH:
        matches = sorted(matches, key=itemgetter(3, 0), reverse=True)

    # Finally, print results in a nice table.
    print(tabulate(matches, headers=['word', 'frequency %', 'occurrences', 'length']))

if __name__ == '__main__':
    main()
