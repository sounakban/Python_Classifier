import multiprocessing
import numpy
import itertools
import math


thread_count = int(multiprocessing.cpu_count()*0.75)

def cal_PMI(cooccurence_list):
    curr_list = cooccurence_list[0]
    curr_vocab = cooccurence_list[1]
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
            curr_list[pair] = math.log(curr_list[pair]/float(curr_vocab[pair[0]]*curr_vocab[pair[1]]))
        else:
            del curr_list[pair]
    return curr_list


def cal_Jaccard(cooccurence_list):
    curr_list = cooccurence_list[0]
    curr_vocab = cooccurence_list[1]
    #Calculate Jaccard Coefficients
    for pair in curr_list.keys():
        if pair[0] in curr_vocab.keys() and pair[1] in curr_vocab.keys() and curr_vocab[pair[0]]+curr_vocab[pair[1]] > curr_list[pair]:
            curr_list[pair] = curr_list[pair]/float(curr_vocab[pair[0]]+curr_vocab[pair[1]]-curr_list[pair])
        else:
            del curr_list[pair]
    return curr_list


def normalize_corcoeff(cooccurence_list):
    curr_list = cooccurence_list
    value_list = curr_list.values()
    mean = float(sum(value_list))/len(value_list)
    for pair in curr_list.keys():
        if curr_list[pair] <= mean:
            curr_list[pair] = 1
        else:
            curr_list[pair] = curr_list[pair]/mean
    return curr_list
