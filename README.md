# ngram-finder

Compare two texts to find common word sequences (n-grams)

Requires Python 3 and the `nltk` module.

On first run, the needed nltk data will be downloaded to its default location.

```
usage: ngram-finder.py [-h] [--min-words MIN_WORDS] [--max-words MAX_WORDS]
                       file1 file2

Compare two texts to find common word sequences (n-grams)

positional arguments:
  file1                 first text file to compare
  file2                 second text file to compare

optional arguments:
  -h, --help            show this help message and exit
  --min-words MIN_WORDS
                        min number of words to look for (default: 4)
  --max-words MAX_WORDS
                        max number of words to look for, must be larger than
                        or equal to min words (default: number of words in the
                        longest sentence found in either text
```