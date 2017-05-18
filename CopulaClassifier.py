from cooccurence_extract import process_text
from math import log, exp


def classify(corcoeff, vocab, test_docs):
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
                    sub_scores.extend(log( phi_inv(phi(curr_vocab[pair[0]], theta) + phi(curr_vocab[pair[1]], theta) , theta) ))
            score = exp(sum(sub_scores))
            scorelist.extend(score)
        prediction = analyze_scores(scorelist, "single")
        predictions.append(prediction)
    print predictions
    return predictions

def bivariate_gumbel(p1, p2, theta):
    return log( phi_inv(phi(p1, theta) + phi(p2, theta), theta) )

def phi_gumbel(termWeight, theta):
    return (-log(termWeight))**theta

def phi_inv_gumbel(termWeight, theta):
    return exp( -(termWeight ** (1/theta)) )

def analyze_scores(scores, predtype):
    if predtype == "single":
        pred = [0]*len(scores)
        pred[scores.index(min(scores))] = 1
    return pred






"""
class CopulaClassifier:
    cooccurences = {}

    def __init__(self, cooccurences):
        self.cooccurences = {tuple(k):v for k,v in cooccurences.items() if len(tuple(k))==2}

    def classify(self, param):
        corcoeff = param[0]
        term_weights = param[1]
        sub_scores = []
        for pair in self.cooccurences.keys():
            if pair in corcoeff.keys():
                theta = corcoeff[pair]
                term_weights[pair[0]]
                term_weights[pair[1]]
                sub_scores.extend(log( phi_inv(phi(term_weights[pair[0]], theta) + phi(term_weights[pair[1]], theta) , theta) ))

        score = exp(sum(sub_scores))
        print score
        return score
"""


"""
class CopulaClassifier:
    cooccurences = {}

    def __init__(self, all_corcoeff, all_term_weight):
        self.all_corcoeff = all_corcoeff
        self.all_term_weight = all_term_weight
        self.

    def classify(self, doc):
        doc_repr = process_text(doc)
        cooccurences = {tuple(k):v for k,v in doc_repr.items() if len(tuple(k))==2}
        sub_scores = []
        for pair in self.cooccurences.keys():
            if pair in corcoeff.keys():
                theta = corcoeff[pair]
                term_weights[pair[0]]
                term_weights[pair[1]]
                sub_scores.extend(log( phi_inv(phi(term_weights[pair[0]], theta) + phi(term_weights[pair[1]], theta) , theta) ))

        score = exp(sum(sub_scores))
        print score
        return score

    def bivariate_gumbel(p1, p2, theta):
        return log( phi_inv(phi(p1, theta) + phi(p2, theta), theta) )

    def phi_gumbel(termWeight, theta):
        return (-log(termWeight))**theta

    def phi_inv_gumbel(termWeight, theta):
        return exp( -(termWeight ** (1/theta)) )
"""
