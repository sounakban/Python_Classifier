from nltk.corpus import stopwords
import string

import glob
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import itertools
import csv


#Return clean sentence
stop = stopwords.words('english') + list(string.punctuation)
def clean_text(sent):
    word_list = [word.lower() for word in word_tokenize(sent) if word.lower() not in stop]
    return word_list


def calc(location):
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
                for pair in word_pairs:
                    if pair in all_pairs:
                        all_pairs[pair] += 1
                    else:
                        all_pairs[pair] = 1

    #print "writing results to file : Cooccur.csv"
    #for key, val in all_pairs.items():
    #    writeFile1.writerow([key, val])
