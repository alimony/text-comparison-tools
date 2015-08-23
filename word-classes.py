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
SORT_CLASS = 'class'

DEFAULT_STOPWORDS = False
DEFAULT_SORT = SORT_OCCURRENCES

# Alphabetical list of part-of-speech tags used in the Penn Treebank Project:
PENN_TREEBANK_POS = {
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'DT': 'Determiner',
    'EX': 'Existential there',
    'FW': 'Foreign word',
    'IN': 'Preposition or subordinating conjunction',
    'JJ': 'Adjective',
    'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative',
    'LS': 'List item marker',
    'MD': 'Modal',
    'NN': 'Noun, singular or mass',
    'NNS': 'Noun, plural',
    'NNP': 'Proper noun, singular',
    'NNPS': 'Proper noun, plural',
    'PDT': 'Predeterminer',
    'POS': 'Possessive ending',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb, comparative',
    'RBS': 'Adverb, superlative',
    'RP': 'Particle',
    'SYM': 'Symbol',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, gerund or present participle',
    'VBN': 'Verb, past participle',
    'VBP': 'Verb, non-3rd person singular present',
    'VBZ': 'Verb, 3rd person singular present',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive wh-pronoun',
    'WRB': 'Wh-adverb',
}


def word_class(word):
    pos_tag = nltk.pos_tag([word])
    tag = pos_tag[0][1]
    word_class = PENN_TREEBANK_POS[tag]

    return word_class


def main():
    parser = argparse.ArgumentParser(description='Categorize (nouns, verbs, etc.) all words in a text')

    parser.add_argument('file', type=argparse.FileType('rU'),
                        help='text file to categorize')

    parser.add_argument('--include-stopwords', action='store_true', default=DEFAULT_STOPWORDS,
                        help='include very common words in results (default: {})'
                        .format(DEFAULT_STOPWORDS))

    parser.add_argument('--sort', type=str, default=DEFAULT_SORT,
                        choices=[SORT_ALPHA, SORT_OCCURRENCES, SORT_LENGTH, SORT_CLASS],
                        help='''how to sort output; by word alphabetically,
                        number of occurrences, word length, or word class (default: {})'''
                        .format(DEFAULT_SORT))

    args = parser.parse_args()

    if not args.file:
        sys.exit('You must specify a text file to categorize')

    # Download required nltk data if needed.
    nltk.download('maxent_treebank_pos_tagger')
    if not args.include_stopwords:
        nltk.download('stopwords')

    # Create tokenized lists of words from the text.
    text = args.file.read()
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Only keep words consisting entirely of letters (i.e. remove punctuation,
    # numbers, etc.)
    tokens = [word for word in tokens if word.isalpha()]

    # Save number of occurrences for each word for later reference, which could
    # as well be through a collections.Counter but FreqDist will cover more
    # cases if expanded on later.
    fdist = FreqDist(tokens)

    # Remove stopwords, i.e. very common ones.
    tokens = set(tokens)
    if not args.include_stopwords:
        tokens -= STOPWORDS

    print('Found {} unique words'.format(len(tokens)))

    # Make a list of (word, occurrences, length, word_class) tuples for final
    # output. `word` is the word itself, `occurrences` the number of times it
    # occurs in its source text, `length` the length of the word, and
    # `word_class` what type of word it is, e.g. noun, verb, etc.
    matches = [(word, fdist.get(word), len(word), word_class(word)) for word in tokens]
    headers = ['word', 'occurrences', 'length', 'word_class']

    # Sort output accordingly.
    if args.sort == SORT_ALPHA:
        matches = sorted(matches, key=itemgetter(0))
    elif args.sort == SORT_OCCURRENCES:
        matches = sorted(matches, key=itemgetter(1, 0), reverse=True)
    elif args.sort == SORT_LENGTH:
        matches = sorted(matches, key=itemgetter(2, 0), reverse=True)
    elif args.sort == SORT_CLASS:
        matches = sorted(matches, key=itemgetter(3, 0))

    # Finally, print results in a nice table.
    print(tabulate(matches, headers=headers))

if __name__ == '__main__':
    main()
