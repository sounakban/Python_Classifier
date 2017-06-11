import multiprocessing
from functools import partial
thread_count = int(multiprocessing.cpu_count()*0.75)
from cooccurence_utils import cal_PMI, get_P_AandB, cal_Jaccard, normalize_corcoeff, get_Simple_Prob, feature_selection



def add_pairs(class_cooc, (pair, value)):
    class_cooc[pair] = class_cooc.get(pair, 0) + value

def get_cooccurences(train_labels, train_docs, P_AandB, term_freq={}, term_prob={}):
    if P_AandB == True and (len(term_freq) != train_labels.shape[1] or len(term_prob) != train_labels.shape[1]):
        raise ValueError('@get_cooccurences: Either set P_AandB to False or pass a valid Term(Freq/Prob) dict, divided by class')

    from cooccurence_extract import process_text
    import numpy

    cooccurence_list = {}
    for i in range(train_labels.shape[1]):
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=100)
        classdocs = [train_docs[did] for did in classdoc_ids]
        file_coocs = pool.imap(process_text, classdocs)
        pool.close()
        pool.join()
        class_cooc = {}
        map(lambda fil_cooc: map(partial(add_pairs, class_cooc), fil_cooc.items()), file_coocs)
        cooccurence_list[i] = class_cooc
    print "All Cooccurences Generated."
    #feature_selection(cooccurence_list)
    #print "Feature Selection Complete"
    #convert from frozenset to tuple
    for i in range(len(cooccurence_list)):
        cooccurence_list[i] = {tuple(k):v for k,v in cooccurence_list[i].items()}
    #If user wants probability of occurence instead of raw-frequency
    if P_AandB == True:
        pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=10)

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
        #"""
        #For feature selection
        cooccurence_list = {i: v for i, v in enumerate(cooccurence_list)}
    feature_selection(cooccurence_list)
    cooccurence_list = list(cooccurence_list.values())
    #print "Feature Selection Complete"
    #"""
    return cooccurence_list


def get_cooccurences_BR(train_labels, train_docs, P_AandB, term_freq={}, term_prob={}):
    if P_AandB == True and (len(term_freq) != 2*train_labels.shape[1] or len(term_prob) != 2*train_labels.shape[1]):
        raise ValueError('@get_cooccurences: Either set P_AandB to False or pass a valid Term(Freq/Prob) dict, divided by class')
    from cooccurence_extract import process_text
    import numpy
    num_class = train_labels.shape[1]

    #Find Cooccurences accross all documents
    pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=100)
    file_coocs = list(pool.imap(process_text, train_docs))
    pool.close()
    pool.join()
    total_cooc = {}
    map(lambda fil_cooc: map(partial(add_pairs, total_cooc), fil_cooc.items()), file_coocs)
    #total_cooc = {k:v for k,v in total_cooc.items() if len(tuple(k))==2}
    print "Training-Corpus Cooccurences Generated."

    #Find cooccurences for each class
    cooccurence_list = {}
    import operator
    for i in range(num_class):
        class_cooc = {}
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        class_files_cooc = map(lambda id: file_coocs[id], classdoc_ids)
        map(lambda fil_cooc: map(partial(add_pairs, class_cooc), fil_cooc.items()), class_files_cooc)
        #class_cooc = dict(sorted(class_cooc.iteritems(), key=operator.itemgetter(1), reverse=True)[:20000])
        cooccurence_list[i] = class_cooc
        cooccurence_list[num_class + i] = dict(total_cooc)
        cooccurence_list[num_class + i].update((k, (cooccurence_list[num_class + i][k] - v)) for k, v in class_cooc.items())
        temp_list = [cooccurence_list[i], cooccurence_list[num_class + i]]
        #print "Initial size: {} | {}".format(len(cooccurence_list[i]), len(cooccurence_list[num_class + i]))
        #feature_selection(temp_list, total_cooc.keys())
        cooccurence_list[i] = temp_list[0]
        cooccurence_list[num_class + i] = temp_list[1]
        #print "Final size: {} | {}".format(len(cooccurence_list[i]), len(cooccurence_list[num_class + i]))
        if (i+1) % 10 == 0:
            print "Processed {} Classes.".format(i+1)
    print "All Cooccurences Generated."
    #Empty memory of unused variables
    file_coocs = []

    #convert from frozenset to tuple
    for i in range(2*num_class):
        cooccurence_list[i] = {tuple(k):v for k,v in cooccurence_list[i].items()}
    #If user wants probability of occurence instead of raw-frequency
    if P_AandB == True:
        pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=10)
        parameter = []
        for i in range(2*num_class):
            parameter.append([cooccurence_list[i], term_freq[i], term_prob[i]])
        cooccurence_list_new = list(pool.map(get_P_AandB, parameter))
        pool.close()
        pool.join()
        cooccurence_list = cooccurence_list_new
    return cooccurence_list





def calc_corcoeff(cooccurence_list, vocab, cor_type, boost = 2):
    from functools import partial

    pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=10)
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
    pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=10)
    new = pool.map(partial(normalize_corcoeff, boost), cooccurence_list)
    cooccurence_list = new
    pool.close()
    pool.join()
    """
    #For feature selection
    cooccurence_list = {i: v for i, v in enumerate(cooccurence_list)}
    print cooccurence_list.keys()
    feature_selection(cooccurence_list)
    cooccurence_list = list(cooccurence_list.values())
    print "Feature Selection Complete"
    #"""
    return cooccurence_list
