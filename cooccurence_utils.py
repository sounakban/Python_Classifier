import numpy
import itertools
import math
from functools import partial



def cal_PMI(input_lists):
    curr_cooccur = input_lists[0]
    curr_vocab = input_lists[1]
    #Convert raw-frequency to probability
    key_list = curr_cooccur.keys()
    value_list = curr_cooccur.values()
    tot = float(sum(value_list))
    value_list_array = numpy.array(value_list)/tot
    value_list = value_list_array.tolist()
    curr_cooccur = dict(itertools.izip(key_list, value_list))
    #Calculate PMI Coefficients
    map(partial(PMI_furmula, curr_cooccur, curr_vocab), key_list)
    return curr_cooccur

def PMI_furmula(curr_cooccur, curr_vocab, pair):
    if pair[0] in curr_vocab and pair[1] in curr_vocab:
        curr_cooccur[pair] = math.log(curr_cooccur[pair]/float(curr_vocab[pair[0]]*curr_vocab[pair[1]]))
    else:
        del curr_cooccur[pair]



def cal_Jaccard(input_lists):
    curr_cooccur = input_lists[0]
    curr_vocab = input_lists[1]
    #Calculate Jaccard Coefficients
    map(partial(Jaccard_formula, curr_cooccur, curr_vocab), curr_cooccur.keys())
    return curr_cooccur

def Jaccard_formula(curr_cooccur, curr_vocab, pair):
    if pair[0] in curr_vocab and pair[1] in curr_vocab:
        divByZerocheck = curr_vocab[pair[0]]+curr_vocab[pair[1]]-curr_cooccur[pair]
        if(divByZerocheck > 0):
            curr_cooccur[pair] = curr_cooccur[pair]/float(divByZerocheck)
    else:
        del curr_cooccur[pair]



def normalize_corcoeff(cooccurence_list):
    curr_list = cooccurence_list
    value_list = curr_list.values()
    mean = float(sum(value_list))/len(value_list)
    for pair in curr_list.keys():
        if curr_list[pair] <= mean:
            curr_list[pair] = 1
        else:
            curr_list[pair] = curr_list[pair]/mean
            if curr_list[pair] > 100.0:
                print pair, curr_list[pair]
    return curr_list
