#!/usr/bin/env python3
# encoding: utf-8

import argparse
import re
import sys

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.util import ngrams
from tabulate import tabulate

DEFAULT_MIN_WORDS = 4


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

    args = parser.parse_args()

    if len(args.files) < 2:
        sys.exit('You must specify at least two text files')

    # Download required nltk data if needed.
    nltk.download('punkt')

    # Create the tokenized lists of words from both texts.
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

    # For every n-gram length we are looking for, make sets and do an
    # intersection to find common ones.
    for i in range(args.min_words, max_words + 1):
        ngram_sets = [set(ngrams(t, i)) for t in tokens]
        matches = set.intersection(*ngram_sets)
        print('Found {} matching n-grams of length {}'.format(len(matches), i))
        all_matches.update(matches)

    print('Found these matches for n-grams of length {} to {}:'.format(args.min_words, max_words))

    # Assemble all matches and sort by word count descending.
    matches = sorted(list(all_matches), key=len, reverse=True)

    # We will remove leading space from punctuation, using the punctuation
    # definition from our sentence tokenizer.
    pattern = ' ([{}])'.format(''.join(PunktSentenceTokenizer.PUNCTUATION))

    # Generate a list of (length, sentence) tuples suited for tabulate.
    lengths_sentences = [(len(m), re.sub(pattern, r'\1', ' '.join(m))) for m in matches]
    lengths_sentences = [(n, len(sentence), sentence) for (n, sentence) in lengths_sentences]

    # Finally, print results in a nice table.
    print(tabulate(lengths_sentences, headers=['n', 'text length', 'full sentence']))

if __name__ == '__main__':
    main()
