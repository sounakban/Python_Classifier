import numpy
import multiprocessing
from cooccurence_extract import process_text
thread_count = multiprocessing.cpu_count()*0.75

from CopulaClassifier import classify


"""
def predict_multiclass(corcoeff, vocab, test_docs):
    doc_classifier = CopulaClassifier(corcoeff, vocab)
    predictions = numpy.array()
    for doc in test_docs:
        class_labels = doc_classifier.classify(doc)
        pool.close()
        pool.join()
        #---add class_labels data to prediction

    return predictions
"""


def predict_multiclass(corcoeff, vocab, test_docs):
    """
    parameter = []
    for i in range(len(corcoeff)):
        parameter.append([corcoeff[i], vocab[i]])
    """
    predictions = numpy.array()
    predictions_list = []

    if thread_count < len(test_docs):
        div = len(test_docs)/thread_count
        processes = []
        for i in range(thread_count):
            end = (i+1)*div-1
            if end >= len(test_docs):
                end = len(test_docs)-1
            processes.append(multiprocessing.Process(target=classify, args=( corcoeff, vocab, test_docs[i*div:end] )))
        for pro in processes:
            predictions_list.extend(pro.start())
        for pro in processes:
            pro.join()
    else:
        predictions_list.extend(classify(corcoeff, vocab, test_docs))
    return predictions

"""
    for doc in test_docs:
        doc_repr = process_text(doc)
        doc_classifier = CopulaClassifier(doc_repr)
        pool = multiprocessing.Pool(processes=thread_count)
        class_labels = pool.imap(doc_classifier.classify, parameter)
        pool.close()
        pool.join()
        #add class_labels data to prdiction
"""
