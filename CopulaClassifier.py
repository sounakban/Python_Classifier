import numpy
from multiprocessing import Process, Queue, cpu_count
thread_count = int(cpu_count()*0.75)

from copula_utils import classify

class CopulaClassifier:

    def __init__(self, corcoeff, vocab):
        self.corcoeff = corcoeff
        self.vocab = vocab

    def predict_multiclass(self, test_docs):

        #predictions = numpy.array()
        predictions_list = []
        que = Queue()

        if thread_count < len(test_docs):
            div = (len(test_docs)/thread_count)+1
            print len(test_docs)
            processes = []
            for i in range(thread_count):
                end = (i+1)*div
                if end > len(test_docs):
                    end = len(test_docs)
                processes.append(Process(target=classify, args=( self.corcoeff, self.vocab, test_docs[i*div:end], que )))
            for pro in processes:
                pro.start()
            for pro in processes:
                predictions_list.extend(que.get())
            for pro in processes:
                pro.join()
        else:
            predictions_list.extend(classify(self.corcoeff, self.vocab, test_docs))
        print(len(predictions_list))
        return numpy.array(predictions_list)
