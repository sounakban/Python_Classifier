from cooccurence_extract import process_text
from math import log, exp
import numpy
import functools


def classify(corcoeff, vocab, priors, test_docs, num, que, predtype):
    if predtype == "multi":
        from Thresholding import M_Cut#, P_Cut
    predictions = [num]
    all_scores = []
    for doc in test_docs:
        scorelist = []
        doc_repr = process_text(doc)
        cooccurences = {tuple(k):v for k,v in doc_repr.items() if len(tuple(k))==2}
        for i in range(len(corcoeff)):
            sub_scores = []
            curr_coeffs = corcoeff[i]
            curr_vocab = vocab[i]
            sub_scores = map(functools.partial(get_scores_one, curr_coeffs, curr_vocab), cooccurences.keys())
            sub_scores = list(filter(None, sub_scores))
            sub_scores = numpy.log(numpy.array(sub_scores))
            #priors = (numpy.array(priors)/min(priors)).tolist()
            score = numpy.sum(sub_scores)#/priors[i]
            #print score, len(sub_scores)
            scorelist.append(score)
        #print scorelist
        if predtype == "single":
            prediction = pred_maxScore(scorelist)
            predictions.append(prediction)
        if predtype == "multi":
            all_scores.append(scorelist)
    if predtype == "multi":
        predictions.extend(M_Cut(all_scores))
    que.put(predictions)




#----------------------Common Functions----------------------

def get_scores_one(curr_coeffs, curr_vocab, pair):
    if pair[0] not in curr_vocab or pair[1] not in curr_vocab:
        return 0.0;
    theta = curr_coeffs.get(pair, 1)
    #print pair, curr_vocab[pair[0]], curr_vocab[pair[1]], curr_coeffs[pair]
    #return bivariate_gumbel(curr_vocab.get(pair[0], 0.1**6), curr_vocab.get(pair[1], 0.1**6), theta)
    return bivariate_gumbel(curr_vocab[pair[0]], curr_vocab[pair[1]], theta)

def bivariate_gumbel(p1, p2, theta):
    res = phi_inv_gumbel(phi_gumbel(p1, theta) + phi_gumbel(p2, theta), theta)
    if res <= 0.0:
        print "PROBLEM: ", p1, p2, theta
    return res

def phi_gumbel(termWeight, theta):
    return (-log(termWeight))**theta

def phi_inv_gumbel(termWeight, theta):
    return exp( -(termWeight ** (1/theta)) )


def pred_maxScore(scores):
    pred = [0]*len(scores)
    pred[scores.index(max(scores))] = 1
    return numpy.array(pred)
