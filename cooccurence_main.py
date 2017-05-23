import multiprocessing
thread_count = int(multiprocessing.cpu_count()*0.75)
from functools import partial
import numpy
import itertools
import math

from cooccurence_extract import process_text
from cooccurence_utils import cal_PMI, cal_Jaccard, normalize_corcoeff




def get_cooccurences(train_labels, train_docs):
    cooccurence_list = {}
    for i in range(train_labels.shape[1]):
        pool = multiprocessing.Pool(processes=thread_count)
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        classdocs = [train_docs[did] for did in classdoc_ids]
        file_coocs = pool.imap(process_text, classdocs)
        pool.close()
        pool.join()
        class_cooc = {}
        map(lambda fil_cooc: map(partial(add_pairs, class_cooc, fil_cooc), fil_cooc.keys()), file_coocs)
        class_cooc_new = {tuple(k):v for k,v in class_cooc.items() if len(tuple(k))==2}
        cooccurence_list[i] = class_cooc_new
    return cooccurence_list

def add_pairs(class_cooc, fil_cooc, pair):
    if pair in class_cooc:
        class_cooc[pair] += fil_cooc[pair]
    else:
        class_cooc[pair] = fil_cooc[pair]




def calc_corcoeff(cooccurence_list, vocab, cor_type):
    pool = multiprocessing.Pool(processes=thread_count)
    parameter = []
    for i in range(len(cooccurence_list)):
        parameter.append([cooccurence_list[i], vocab[i]])
    if cor_type == "P":
        cooccurence_list_new = list(pool.imap(cal_PMI, parameter))
    elif cor_type == "J":
        cooccurence_list_new = list(pool.imap(cal_Jaccard, parameter))
    pool.close()
    pool.join()
    cooccurence_list = cooccurence_list_new

    pool = multiprocessing.Pool(processes=thread_count)
    new = pool.map(normalize_corcoeff, cooccurence_list)
    cooccurence_list = new
    pool.close()
    pool.join()
    return cooccurence_list
