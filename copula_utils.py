from cooccurence_extract import process_text
from math import log, exp
import numpy
import functools



#----------------------Global Functions----------------------

def classify(corcoeff, vocab, test_docs, num, que, predtype):
    predictions = [num]
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
            """
            for pair in cooccurences.keys():
                if pair in curr_coeffs:
                    theta = curr_coeffs[pair]
                    #sub_scores.append(log( phi_inv_gumbel(phi_gumbel(curr_vocab[pair[0]], theta) + phi_gumbel(curr_vocab[pair[1]], theta) , theta) ))
                    sub_scores.append( bivariate_gumbel(curr_vocab[pair[0]], curr_vocab[pair[1]], theta) )
            """
            score = sum(sub_scores)
            #score = exp(sum(sub_scores))
            scorelist.append(score)
        prediction = score2pred(scorelist, predtype)
        predictions.append(prediction)
        print "Doc complete"
    que.put(predictions)



def classify_OnevsAll(corcoeff, vocab, test_docs, num, que, predtype):
    predictions = [num]
    for doc in test_docs:
        doc_repr = process_text(doc)
        cooccurences = {tuple(k):v for k,v in doc_repr.items() if len(tuple(k))==2}
        prediction = []
        for i in range(len(corcoeff)):
            sub_scores = []
            curr_coeffs = corcoeff[i]
            curr_vocab = vocab[i]
            sub_scores_one = map(functools.partial(get_scores_one, curr_coeffs, curr_vocab), cooccurences.keys())
            sub_scores_one = filter(None, sub_scores)
            sub_scores_all = map(functools.partial(get_scores_all, corcoeff, vocab, i), cooccurences.keys())
            sub_scores_all = filter(None, sub_scores)
            score_one = numpy.log(numpy.array(sub_scores_one))
            score_one = numpy.sum(sub_scores_one)
            score_all = numpy.log(numpy.array(sub_scores_all))
            score_all = numpy.sum(sub_scores_all)
            if score_one < score_all:
                prediction.append(1)
            else:
                prediction.append(0)
        predictions.append(prediction)
        print "Doc complete"
    que.put(predictions)







#----------------------Common Functions----------------------

def get_scores_one(curr_coeffs, curr_vocab, pair):
    if pair in curr_coeffs:
        theta = curr_coeffs[pair]
        return bivariate_gumbel(curr_vocab[pair[0]], curr_vocab[pair[1]], theta)

def get_scores_all(corcoeff, vocab, i, pair):
    theta = 0.0
    p1 = 0.0
    p2 = 0.0
    categories = len(corcoeff)
    for j in range(categories):
        if pair in corcoeff[j]:
            if not i == j:
                p1 += vocab[j][pair[0]]
                p2 += vocab[j][pair[1]]
                theta += corcoeff[j][pair]
    theta = theta/categories
    if theta >= 1.0:
        p1 = p1/categories
        p2 = p2/categories
        return bivariate_gumbel(p1, p2, theta)


def bivariate_gumbel(p1, p2, theta):
    if phi_inv_gumbel(phi_gumbel(p1, theta) + phi_gumbel(p2, theta), theta) <= 0.0:
        print "PROBLEM: ", p1, p2, theta
    return phi_inv_gumbel(phi_gumbel(p1, theta) + phi_gumbel(p2, theta), theta)

def phi_gumbel(termWeight, theta):
    return (-log(termWeight))**theta

def phi_inv_gumbel(termWeight, theta):
    return exp( -(termWeight ** (1/theta)) )


def score2pred(scores, predtype):
    if predtype == "single":
        pred = [0]*len(scores)
        pred[scores.index(max(scores))] = 1
    elif predtype == "multi":
        list_mean = numpy.mean(numpy.array(scores))
        st_dev = numpy.std(numpy.array(scores))
        thresh = list_mean-(2*st_dev)
        print "Threshold: ", thresh, list_mean, st_dev
        pred = [0]*len(scores)
        classes = [i for i, x in enumerate(scores) if x <= thresh]
        for i in classes:
            pred[i] = 1
    return numpy.array(pred)
