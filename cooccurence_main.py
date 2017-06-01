import multiprocessing
thread_count = int(multiprocessing.cpu_count()*0.75)
from cooccurence_utils import cal_PMI, get_P_AandB, cal_Jaccard, normalize_corcoeff, get_Simple_Prob, feature_selection




def get_cooccurences(train_labels, train_docs, P_AandB, term_freq={}, term_prob={}):
    if P_AandB == True and (len(term_freq) != train_labels.shape[1] or len(term_prob) != train_labels.shape[1]):
        raise ValueError('@get_cooccurences: Either set P_AandB to False or pass a valid Term(Freq/Prob) dict, divided by class')

    from cooccurence_extract import process_text
    from functools import partial
    import numpy

    cooccurence_list = {}
    for i in range(train_labels.shape[1]):
        pool = multiprocessing.Pool(processes=thread_count)
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        classdocs = [train_docs[did] for did in classdoc_ids]
        file_coocs = pool.imap(process_text, classdocs)
        pool.close()
        pool.join()
        class_cooc = {}
        map(lambda fil_cooc: map(partial(add_pairs, class_cooc, fil_cooc), fil_cooc.keys()), file_coocs)
        class_cooc_new = {k:v for k,v in class_cooc.items() if len(tuple(k))==2}
        cooccurence_list[i] = class_cooc_new
    #feature_selection(cooccurence_list)
    #convert from frozenset to tuple
    for i in range(len(cooccurence_list)):
        cooccurence_list[i] = {tuple(k):v for k,v in cooccurence_list[i].items()}
    #If user wants probability of occurence instead of raw-frequency
    if P_AandB == True:
        pool = multiprocessing.Pool(processes=thread_count)

        #Option 1:
        #cooccurence_list_new = list(pool.map(get_Simple_Prob, cooccurence_list.values()))

        #Option 2:
        parameter = []
        for i in range(len(cooccurence_list)):
            parameter.append([cooccurence_list[i], term_freq[i], term_prob[i]])
        cooccurence_list_new = list(pool.map(get_P_AandB, parameter))

        pool.close()
        pool.join()
        cooccurence_list = cooccurence_list_new
        """
        #For feature selection
        cooccurence_list = {i: v for i, v in enumerate(cooccurence_list)}
    feature_selection(cooccurence_list)
    cooccurence_list = list(cooccurence_list.values())
    """
    return cooccurence_list

def add_pairs(class_cooc, fil_cooc, pair):
    if pair in class_cooc:
        class_cooc[pair] += fil_cooc[pair]
    else:
        class_cooc[pair] = fil_cooc[pair]




def calc_corcoeff(cooccurence_list, vocab, cor_type):
    pool = multiprocessing.Pool(processes=thread_count)
    parameter = []
    for i in range(len(cooccurence_list)):
        parameter.append([cooccurence_list[i], vocab[i]])
    if cor_type == "P":
        cooccurence_list_new = list(pool.imap(cal_PMI, parameter))
    elif cor_type == "J":
        cooccurence_list_new = list(pool.imap(cal_Jaccard, parameter))
    pool.close()
    pool.join()
    cooccurence_list = cooccurence_list_new
    pool = multiprocessing.Pool(processes=thread_count)
    new = pool.map(normalize_corcoeff, cooccurence_list)
    cooccurence_list = new
    """
    #For feature selection
    cooccurence_list = {i: v for i, v in enumerate(cooccurence_list)}
    feature_selection(cooccurence_list)
    cooccurence_list = list(cooccurence_list.values())
    """
    pool.close()
    pool.join()
    return cooccurence_list
