import numpy
import itertools
import math
from functools import partial



def cal_PMI(input_lists):
    curr_cooccur = input_lists[0]
    curr_vocab = input_lists[1]
    #Calculate PMI Coefficients
    map(partial(PMI_furmula, curr_cooccur, curr_vocab), curr_cooccur.keys())
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
            print "Cooccurence too high for term-pair {}, w1: {}, w2: {}, freq: {}".format(pair, curr_vocab[pair[0]], curr_vocab[pair[1]], curr_cooccur[pair])
            del curr_cooccur[pair]
    else:
        del curr_cooccur[pair]



def normalize_corcoeff(cooccurence_list):
    curr_list = cooccurence_list
    value_list = curr_list.values()
    mean = float(sum(value_list))/len(value_list)
    #print curr_list.values()
    #print len(value_list), mean
    for pair in curr_list.keys():
        if curr_list[pair] <= mean:
            curr_list[pair] = 1
        else:
            curr_list[pair] = curr_list[pair]/mean
            if curr_list[pair] > 100.0:
                print "@normalize: ", pair, curr_list[pair]
    return curr_list



def get_P_AandB(parameter):
    cooccur = parameter[0]
    vocab_tf = parameter[1]
    vocab_tprob = parameter[2]
    map(partial(P_AandB_formula, cooccur, vocab_tf, vocab_tprob), cooccur.keys())
    return cooccur

def P_AandB_formula(cooccur, vocab_tf, vocab_tprob, pair):
    if pair[0] in vocab_tf and pair[1] in vocab_tf:
        P_BgivenA = cooccur[pair]/float(vocab_tf[pair[0]])
        if(P_BgivenA > 1):
            print "P_BgivenA > 1, @ get_P_AandB"
        P_AandB = vocab_tprob[pair[0]] * P_BgivenA
        if(P_AandB > 1):
            print "P_AandB > 1, @ get_P_AandB"
            del cooccur[pair]
        elif(P_AandB == 0.0):
            print pair, ": {}/{}= {}. {}".format(cooccur[pair], vocab_tf[pair[0]], P_BgivenA, vocab_tprob[pair[0]])
        else:
            cooccur[pair] = P_AandB
    else:
        del cooccur[pair]
