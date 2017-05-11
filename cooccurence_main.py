import multiprocessing
from functools import partial
import numpy
import itertools
import math

from cooccurence_extract import process_text
from cooccurence_utils import cal_PMI, cal_Jaccard, normalize_corcoeff


thread_count = (multiprocessing.cpu_count()*75)/100


def get_cooccurences(train_labels, train_docs):
    cooccurence_list = {}
    for i in range(train_labels.shape[1]):
        pool = multiprocessing.Pool(processes=thread_count)
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        classdocs = [train_docs[did] for did in classdoc_ids]
        file_coocs = pool.imap_unordered(process_text, classdocs)
        pool.close()
        pool.join()
        class_cooc = {}
        for fil_cooc in file_coocs:
            for pair in fil_cooc.keys():
                if pair in class_cooc:
                    class_cooc[pair] += fil_cooc[pair]
                else:
                    class_cooc[pair] = fil_cooc[pair]
        cooccurence_list[i] = class_cooc
    return cooccurence_list


def cacl_corcoff(cooccurence_list, vocab, cor_type):
    pool = multiprocessing.Pool(processes=thread_count)
    parameter = []
    for i in range(len(cooccurence_list)):
        parameter.append([cooccurence_list[i], vocab[i]])
    #print len(parameter)
    if cor_type == "P" or cor_type == "p":
        pool.imap(cal_PMI, parameter)
    elif cor_type == "J" or cor_type == "j":
        pool.imap(cal_Jaccard, parameter)
    pool.close()
    pool.join()

    pool = multiprocessing.Pool(processes=thread_count)
    norm_func = partial(normalize_corcoeff, cooccurence_list)
    pool.map(norm_func, range(len(cooccurence_list)))
    pool.close()
    pool.join()
