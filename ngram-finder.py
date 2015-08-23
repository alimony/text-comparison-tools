#!/usr/bin/env python3
# encoding: utf-8

import argparse
import re
import sys
from itertools import groupby
from operator import itemgetter

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.util import ngrams
from tabulate import tabulate

SORT_N = 'n'
SORT_LENGTH = 'length'
SORT_ALPHA = 'alpha'

DEFAULT_MIN_WORDS = 4
DEFAULT_ZERO_RESULTS = 3
DEFAULT_PUNCTUATION = '()' + ''.join(PunktSentenceTokenizer.PUNCTUATION)
DEFAULT_INCLUDE_SUBGRAMS = False
DEFAULT_SORT = SORT_N


def main():
    parser = argparse.ArgumentParser(description='Compare texts to find common word sequences (n-grams)')

    parser.add_argument('files', nargs='+', type=argparse.FileType('rU'),
                        help='text files to compare')

    parser.add_argument('--min-words', type=int, default=DEFAULT_MIN_WORDS,
                        help='min number of words to look for (default: {})'
                        .format(DEFAULT_MIN_WORDS))

    parser.add_argument('--max-words', type=int,
                        help='''max number of words to look for, must be larger than
                             or equal to min words (default: number of words in the
                             longest sentence found in either text''')

    parser.add_argument('--zero-results', type=int, default=DEFAULT_ZERO_RESULTS,
                        help='''stop comparison after this many zero results for
                             next n-gram length, set to 0 to disable (default: {})'''
                             .format(DEFAULT_ZERO_RESULTS))

    parser.add_argument('--punctuation', type=str, default=DEFAULT_PUNCTUATION,
                        help='''what characters should count as punctuation (default: {})'''
                        .format(DEFAULT_PUNCTUATION))

    parser.add_argument('--include-subgrams', action='store_true', default=DEFAULT_INCLUDE_SUBGRAMS,
                        help='''skip resulting n-grams that are subsets of any
                             longer n-grams found (default: {})'''
                             .format(DEFAULT_INCLUDE_SUBGRAMS))

    parser.add_argument('--sort', type=str, default=DEFAULT_SORT, choices=[SORT_N, SORT_LENGTH, SORT_ALPHA],
                        help='''how to sort final output, by n-gram length, text
                             length, or sentence alphabetically (default: {})'''
                             .format(DEFAULT_SORT))

    args = parser.parse_args()

    if len(args.files) < 2:
        sys.exit('You must specify at least two text files to compare')

    # Download required nltk data if needed.
    nltk.download('punkt')

    # Build list of what should count as punctuation.
    punctuation_set = set(args.punctuation)

    # Create the tokenized lists of words from all texts.
    texts = [f.read() for f in args.files]
    texts = [t.lower() for t in texts]
    tokens = [nltk.word_tokenize(t) for t in texts]

    print('Setting min words to {}'.format(args.min_words))

    # If max words is not passed as a parameter, find out what the longest
    # sentence is in either text and use that number of words as max.
    if args.max_words:
        max_words = args.max_words
        print('Setting max words to {}'.format(max_words))
    else:
        sentences = sum([nltk.sent_tokenize(t) for t in texts], [])
        word_counts = [len(nltk.word_tokenize(s)) for s in sentences]
        max_words = max(word_counts)
        print('Setting max words to {} based on longest sentence found'.format(max_words))

    if max_words < args.min_words:
        sys.exit('Max words must be equal to or larger than min words, exiting')

    all_matches = set()
    zero_results = 0

    # For every n-gram length we are looking for, make sets and do an
    # intersection to find common ones.
    for i in range(args.min_words, max_words + 1):
        print('Checking for n-grams of length {}'.format(i))

        ngram_sets = [set(ngrams(t, i)) for t in tokens]
        matches = set.intersection(*ngram_sets)

        # Filter out any matching n-grams that are less than min_words long when
        # disregarding punctuation, add them to their respective set instead.
        true_matches = []
        count = 0
        for m in matches:
            true_n = len(set(m) - punctuation_set)
            if true_n == i:
                count += 1
            if true_n >= args.min_words:
                true_matches.append((true_n, m))

        all_matches.update(true_matches)

        if args.zero_results > 0 and count == 0:
            zero_results += 1
            if zero_results >= args.zero_results:
                print('Got zero results for {} consecutive n-gram lengths, continuing'.format(args.zero_results))
                break

    if not all_matches:
        print('Found no matches, exiting')
        sys.exit()

    # Filter out n-grams that are part of a longer n-gram in the results.
    if not args.include_subgrams:
        remove = []
        for (n1, m1) in all_matches:
            for (n2, m2) in all_matches:
                if ' '.join(m1) in ' '.join(m2) and n2 > n1:
                    remove.append((n1, m1))
        if remove:
            all_matches -= set(remove)

    counts = {}
    for (true_n, m) in all_matches:
        if true_n not in counts:
            counts[true_n] = 0
        counts[true_n] += 1

    for length, count in sorted(counts.items(), key=itemgetter(0)):
        print('Found {} matches for n-grams of length {}'.format(count, length))

    # We will remove leading space from punctuation, using the punctuation
    # definition from our sentence tokenizer.
    pattern = ' ([{}])'.format(''.join(PunktSentenceTokenizer.PUNCTUATION))

    # Generate a list of (n, length, sentence) tuples suited for tabulate.
    lengths_sentences = [(n, re.sub(pattern, r'\1', ' '.join(m))) for (n, m) in all_matches]
    lengths_sentences = [(n, len(sentence), sentence) for (n, sentence) in lengths_sentences]

    # Sort output accordingly.
    if args.sort == SORT_N:
        lengths_sentences = sorted(lengths_sentences, key=itemgetter(0), reverse=True)
    elif args.sort == SORT_LENGTH:
        lengths_sentences = sorted(lengths_sentences, key=itemgetter(1, 2), reverse=True)
    elif args.sort == SORT_ALPHA:
        lengths_sentences = sorted(lengths_sentences, key=itemgetter(2, 2))

    # Finally, print results in a nice table, or actually one table for each
    # n-gram length for good visual separation.
    for n, results in groupby(lengths_sentences, key=itemgetter(0)):
        print('')
        print(tabulate(results, headers=('n', 'length', 'sentence')))

if __name__ == '__main__':
    main()
