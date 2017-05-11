from nltk.corpus import stopwords
import string

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import itertools
import time
import csv


def cooccurence_main(train_labels, train_docs):
    for i in range(train_labels.shape[1]):
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        
