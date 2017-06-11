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
        cooccurence_list = {i: v for i, v in enumerate(cooccurence_list)}
    #"""
    #Perform feature selection
    feature_selection(cooccurence_list)
    print "Feature Selection Complete"
    #"""
    cooccurence_list = list(cooccurence_list.values())
    return cooccurence_list



def get_cooccurences_BR(train_labels, train_docs, P_AandB, term_freq={}, term_prob={}):
    if P_AandB == True and (len(term_freq) != 2*train_labels.shape[1] or len(term_prob) != 2*train_labels.shape[1]):
        raise ValueError('@get_cooccurences: Either set P_AandB to False or pass a valid Term(Freq/Prob) dict, divided by class')
    from cooccurence_extract import process_text
    import itertools
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
    for i in range(num_class):
        class_cooc = {}
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        class_files_cooc = map(lambda id: file_coocs[id], classdoc_ids)
        map(lambda fil_cooc: map(partial(add_pairs, class_cooc), fil_cooc.items()), class_files_cooc)

        #Generate complement class statictics by list subtraction
        l1 = numpy.array(map(lambda pair: class_cooc.get(pair, 0), total_cooc.keys()))
        l2 = numpy.array(total_cooc.values())
        l2 = (l2 - l1).tolist()
        l1 = l1.tolist()
        cooccurence_list[i] = l1
        cooccurence_list[num_class + i] = l2
        if (i+1) % 10 == 0:
            print "Processed {} Classes.".format(i+1)
    print "All Cooccurences Generated."
    #Empty memory of unused variables
    file_coocs = []

    #convert from frozenset to tuple
    total_cooc = {tuple(k):v for k, v in total_cooc.items()}

    cooccurence_list_new = {}
    for i in range(num_class):
        temp = {}
        temp[i] = dict( filter(lambda x: x[1]>0, itertools.izip(total_cooc.keys(), cooccurence_list[i])) )
        cooccurence_list[i] = []
        temp[num_class + i] = dict( filter(lambda x: x[1]>0, itertools.izip(total_cooc.keys(), cooccurence_list[num_class + i])) )
        cooccurence_list[num_class + i] = []
        temp_new = temp.values()
        #"""
        feature_selection(temp_new, total_cooc.keys())
        temp[i] = temp_new[0]
        temp[num_class + i] = temp_new[1]
        #"""
        if P_AandB == True:
            parameter = []
            parameter.append([temp[i], term_freq[i], term_prob[i]])
            parameter.append([temp[num_class + i], term_freq[num_class + i], term_prob[num_class + i]])
            pool = multiprocessing.Pool(processes=2)
            temp_new = list(pool.map(get_P_AandB, parameter))
            pool.close()
            pool.join()
        #feature_selection(temp_new, total_cooc.keys())
        cooccurence_list_new[i] = temp_new[0]
        cooccurence_list_new[num_class + i] = temp_new[1]
    cooccurence_list = cooccurence_list_new

    #convert from frozenset to tuple
    for i in range(2*num_class):
        cooccurence_list[i] = {tuple(k):v for k,v in cooccurence_list[i].items()}
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
