from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import re
from functools import partial
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import itertools
#import csv


#Return clean sentence
stop = stopwords.words('english') + list(string.punctuation)
def clean_text(sent, word_length = 3):
    min_length = word_length
    words = [word.lower() for word in word_tokenize(sent) if word.lower() not in stop]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens


def process_text(text):
    all_pairs = {}
    #writeFile1 = csv.writer(open('Cooccur.csv', 'w'))

    sent_tokenize_list = sent_tokenize(text)
    for sent in sent_tokenize_list:
        words = clean_text(sent)
        #Eliminate duplicate words to get entence level cooc
        words = list(set(words))
        word_pairs = list(itertools.combinations(words, 2))
        map(partial(add_to_dict, all_pairs), word_pairs)
    #print "writing results to file : Cooccur.csv"
    #for key, val in all_pairs.items():
    #    writeFile1.writerow([key, val])
    return all_pairs


def process_dir(location):
    import glob
    files = glob.glob(location)
    all_pairs = {}
    #writeFile1 = csv.writer(open('Cooccur.csv', 'w'))

    # iterate over the list getting each file
    for fle in files:
        # open the file and then call .read() to get the text \
        with open(fle) as f:
            text = f.read()
            sent_tokenize_list = sent_tokenize(text)
            for sent in sent_tokenize_list:
                words = clean_text(sent)
                word_pairs = list(itertools.combinations(words, 2))
                map(partial(add_to_dict, all_pairs), word_pairs)
    #print "writing results to file : Cooccur.csv"
    #for key, val in all_pairs.items():
    #    writeFile1.writerow([key, val])



def add_to_dict(all_pairs, pair):
    pair_set = frozenset(pair)
    if pair_set in all_pairs:
        all_pairs[pair_set] += 1
    else:
        all_pairs[pair_set] = 1
