# Text Comparison Tools

Scripts to find out what texts have in common.

Requires Python 3 and the `nltk` and `tabulate` modules.

On first run, the needed nltk data will be downloaded to its default location.


### ngram-finder.py

```
usage: ngram-finder.py [-h] [--min-words MIN_WORDS] [--max-words MAX_WORDS]
                       [--zero-results ZERO_RESULTS]
                       [--punctuation PUNCTUATION] [--include-subgrams]
                       [--sort {n,length,alpha}]
                       files [files ...]

Compare texts to find common word sequences (n-grams)

positional arguments:
  files                 text files to compare

optional arguments:
  -h, --help            show this help message and exit
  --min-words MIN_WORDS
                        min number of words to look for (default: 4)
  --max-words MAX_WORDS
                        max number of words to look for, must be larger than
                        or equal to min words (default: number of words in the
                        longest sentence found in either text
  --zero-results ZERO_RESULTS
                        stop comparison after this many zero results for next
                        n-gram length, set to 0 to disable (default: 3)
  --punctuation PUNCTUATION
                        what characters should count as punctuation (default:
                        ();:,.!?)
  --include-subgrams    skip resulting n-grams that are subsets of any longer
                        n-grams found (default: False)
  --sort {n,length,alpha}
                        how to sort final output, by n-gram length, text
                        length, or sentence alphabetically (default: n)
```


### common-words.py

```
usage: common-words.py [-h] [--include-stopwords]
                       [--word-frequency-list WORD_FREQUENCY_LIST]
                       [--sort {alpha,frequency,occurrences,length}]
                       files [files ...]

Compare texts to find common words and their frequencies

positional arguments:
  files                 text files to compare

optional arguments:
  -h, --help            show this help message and exit
  --include-stopwords   include very common words in results (default: False)
  --word-frequency-list WORD_FREQUENCY_LIST
                        providing a text file containing "word;frequency"
                        lines will print that frequency data next to the word
                        in output, and sort on it
  --sort {alpha,frequency,occurrences,length}
                        how to sort output; by word alphabetically, by its
                        frequency in the given texts, number of occurrences,
                        or by word length (default: occurrences)
```
