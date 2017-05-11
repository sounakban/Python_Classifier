import multiprocessing
import numpy
import itertools
import math


thread_count = (multiprocessing.cpu_count()*75)/100

def cal_PMI(cooccurence_list, vocab, i):
    curr_list = cooccurence_list[i]
    curr_vocab = vocab[i]
    #Convert raw-frequency to probability
    key_list = curr_list.keys()
    value_list = curr_list.values()
    tot = float(sum(value_list))
    value_list_array = numpy.array(value_list)/tot
    value_list = value_list_array.tolist()
    curr_list = dict(itertools.izip(key_list, value_list))
    #Calculate PMI Coefficients
    for pair in curr_list.keys():
        if pair[0] in curr_vocab.keys() and pair[1] in curr_vocab.keys():
            if curr_list[pair] == 0 or curr_vocab[pair[0]] == 0 or curr_vocab[pair[1]] == 0:
                del curr_list[pair]
                continue
            curr_list[pair] = math.log(curr_list[pair]/(curr_vocab[pair[0]]*curr_vocab[pair[1]]))
        else:
            del curr_list[pair]


def cal_Jaccard(cooccurence_list, vocab, i):
    curr_list = cooccurence_list[i]
    curr_vocab = vocab[i]
    #Calculate Jaccard Coefficients
    for pair in curr_list.keys():
        if pair[0] in curr_vocab.keys() and pair[1] in curr_vocab.keys():
            if curr_list[pair] == 0 or curr_vocab[pair[0]] == 0 or curr_vocab[pair[1]] == 0:
                del curr_list[pair]
                continue
            curr_list[pair] = curr_list[pair]/(curr_vocab[pair[0]]+curr_vocab[pair[1]]-curr_list[pair])
        else:
            del curr_list[pair]


def normalize_corcoeff(cooccurence_list, i):
    curr_list = cooccurence_list[i]
    value_list = curr_list.values()
    mean = float(sum(value_list))/len(value_list)
    for pair in curr_list.keys():
        if curr_list[pair] <= mean:
            curr_list[pair] = 1
        else:
            curr_list[pair] = curr_list[pair]/mean
