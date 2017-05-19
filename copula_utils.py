from cooccurence_extract import process_text
from math import log, exp
import numpy


def classify(corcoeff, vocab, test_docs, que):
    predictions = []
    for doc in test_docs:
        scorelist = []
        doc_repr = process_text(doc)
        cooccurences = {tuple(k):v for k,v in doc_repr.items() if len(tuple(k))==2}
        for i in range(len(corcoeff)):
            sub_scores = []
            curr_coeffs = corcoeff[i]
            curr_vocab = vocab[i]
            for pair in cooccurences.keys():
                if pair in curr_coeffs.keys():
                    theta = curr_coeffs[pair]
                    #sub_scores.append(log( phi_inv_gumbel(phi_gumbel(curr_vocab[pair[0]], theta) + phi_gumbel(curr_vocab[pair[1]], theta) , theta) ))
                    sub_scores.append( bivariate_gumbel(curr_vocab[pair[0]], curr_vocab[pair[1]], theta) )
            #score = exp(sum(sub_scores))
            score = sum(sub_scores)
            scorelist.append(score)
        print "Scores: ", scorelist
        prediction = analyze_scores(scorelist, "single")
        predictions.append(prediction)
        print "Doc complete"
    #for pred in predictions:
        #print pred
    que.put(predictions)
    #return predictions

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
    return numpy.array(pred)
