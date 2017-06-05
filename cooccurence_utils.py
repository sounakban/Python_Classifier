import numpy
import itertools
import math
from functools import partial





################################## For finding Cooccurences ##################################

def get_Simple_Prob(cooccur):
    tot = float(numpy.sum(numpy.array(cooccur.values())))
    values = (numpy.array(cooccur.values())/tot).tolist()
    res = dict(itertools.izip(cooccur.keys(), values))
    return res


def get_P_AandB(parameter):
    cooccur = parameter[0]
    vocab_tf = parameter[1]
    vocab_tprob = parameter[2]
    map(partial(P_AandB_formula, cooccur, vocab_tf, vocab_tprob), cooccur.keys())
    return cooccur

def P_AandB_formula(cooccur, vocab_tf, vocab_tprob, pair):
    if pair[0] in vocab_tf and pair[1] in vocab_tf:
        P_BgivenA = cooccur[pair]/float(vocab_tf[pair[0]])
        #if(P_BgivenA > 1):
            #print "P_BgivenA > 1, @ get_P_AandB"
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

def feature_selection(cooccurence_list):
    import operator
    #"""
    #Select features using chi-squared test
    all_pairs = []
    for coouc in cooccurence_list.values():
        all_pairs.extend(coouc.keys())
    #Remove duplicates
    all_pairs = list(set(all_pairs))
    #Create matrix for chi-squared test
    matrix = numpy.zeros(shape=(len(cooccurence_list),len(all_pairs)))
    for i in range(len(cooccurence_list)):
        matrix[i] = map(lambda pair: cooccurence_list[i].get(pair, 0), all_pairs)
    #Perform test
    from sklearn.feature_selection import SelectKBest, chi2
    selector = SelectKBest(chi2, k=len(cooccurence_list)*1500)
    matrix = selector.fit_transform(matrix, range(len(cooccurence_list)))
    kept_features = selector.get_support(indices=True)
    all_pairs = [all_pairs[i] for i in kept_features]
    for i in range(len(cooccurence_list)):
        cooccurence_list[i] = {pair: cooccurence_list[i].get(pair, 0) for pair in all_pairs}
        cooccurence_list[i] = {k: v for k, v in cooccurence_list[i].items() if v!=0}
    #"""
    #"""
    #Select top N candidates
    for i in range(len(cooccurence_list)):
        if len(cooccurence_list[i]) > 500:
            cooccurence_list[i] = dict(sorted(cooccurence_list[i].iteritems(), key=operator.itemgetter(1), reverse=True)[:500])
    #"""





################################## For Correlation-Coefficients ##################################

def cal_PMI(input_lists):
    curr_cooccur = input_lists[0]
    curr_vocab = input_lists[1]
    #Calculate PMI Coefficients
    map(partial(PMI_furmula, curr_cooccur, curr_vocab), curr_cooccur.keys())
    return curr_cooccur

def PMI_furmula(curr_cooccur, curr_vocab, pair):
    if pair[0] in curr_vocab and pair[1] in curr_vocab:
        ValueCheck = curr_cooccur[pair]/float(curr_vocab[pair[0]]*curr_vocab[pair[1]])
        #if ValueCheck < 2:
            #print curr_cooccur[pair], curr_vocab[pair[0]], curr_vocab[pair[1]]
        curr_cooccur[pair] = math.log(ValueCheck, 2)
    else:
        del curr_cooccur[pair]
        math.log(x, base=None)



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
    for pair in curr_list.keys():
        if curr_list[pair] <= mean:
            curr_list[pair] = 1
            #curr_list[pair] = curr_list[pair]/mean
        else:
            curr_list[pair] = (curr_list[pair]/mean)**2
            if curr_list[pair] > 100.0:
                print "@normalize: ", pair, curr_list[pair]
    return curr_list
