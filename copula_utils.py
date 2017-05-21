from cooccurence_extract import process_text
from math import log, exp
import numpy
import functools


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
            sub_scores = map(functools.partial(get_scores, curr_coeffs, curr_vocab), cooccurences.keys())
            sub_scores = filter(None, sub_scores)
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
        print "Scores: ", scorelist
        prediction = analyze_scores(scorelist, predtype)
        predictions.append(prediction)
        print "Doc complete"
    #for pred in predictions:
        #print pred
    que.put(predictions)
    #return predictions




def get_scores(curr_coeffs, curr_vocab, pair):
    if pair in curr_coeffs:
        theta = curr_coeffs[pair]
        return bivariate_gumbel(curr_vocab[pair[0]], curr_vocab[pair[1]], theta)

def bivariate_gumbel(p1, p2, theta):
    return log( phi_inv_gumbel(phi_gumbel(p1, theta) + phi_gumbel(p2, theta), theta) )

def phi_gumbel(termWeight, theta):
    return (-log(termWeight))**theta

def phi_inv_gumbel(termWeight, theta):
    return exp( -(termWeight ** (1/theta)) )




def analyze_scores(scores, predtype):
    if predtype == "single":
        pred = [0]*len(scores)
        pred[scores.index(min(scores))] = 1
    elif predtype == "multi":
        list_mean = numpy.mean(numpy.array(scores))
        st_dev = numpy.std(numpy.array(scores))
        thresh = list_mean-(2*st_dev)
        print "Threshold: ", thresh, list_mean, st_dev
        pred = [0]*len(scores)
        classes = [i for i, x in enumerate(scores) if x <= thresh]
        for i in classes:
            pred[i] = 1
        """
        pred = [0]*len(scores)
        classes = [i for i, x in enumerate(scores) if x == 0.0]
        for i in classes:
            pred[i] = 1
        """
    return numpy.array(pred)
