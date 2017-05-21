import numpy
from multiprocessing import Process, Queue, cpu_count
thread_count = int(cpu_count()*0.75)

from copula_utils import classify

class CopulaClassifier:

    def __init__(self, corcoeff, vocab):
        self.corcoeff = corcoeff
        self.vocab = vocab



    def predict_multiclass(self, test_docs):

        predictions_list = []
        predictions_dict = {}
        que = Queue()

        if thread_count < len(test_docs):
            div = (len(test_docs)/thread_count)+1
            processes = []
            for i in range(thread_count):
                end = (i+1)*div
                if end > len(test_docs):
                    end = len(test_docs)
                processes.append(Process(target=classify, args=( self.corcoeff, self.vocab, test_docs[i*div:end], i, que, "single" )))
            for pro in processes:
                pro.start()
            for pro in processes:
                temp = que.get()
                #print "LENGTH OF PREDICTIONS: ", len(temp[1:])
                #print "TYPE OF PREDICTIONS: ", type(temp[1:])
                predictions_dict[temp[0]] = temp[1:]
            for pro in processes:
                pro.join()
            for i in range(len(processes)):
                predictions_list.extend(predictions_dict[i])
        else:
            classify(self.corcoeff, self.vocab, test_docs, 0, que, "single")
            predictions_list.extend(que.get()[1:])
        #print(len(predictions_list))
        return numpy.array(predictions_list)





    def predict_multilabel(self, test_docs):

        predictions_list = []
        predictions_dict = {}
        que = Queue()

        if thread_count < len(test_docs):
            div = (len(test_docs)/thread_count)+1
            processes = []
            for i in range(thread_count):
                end = (i+1)*div
                if end > len(test_docs):
                    end = len(test_docs)
                processes.append(Process(target=classify, args=( self.corcoeff, self.vocab, test_docs[i*div:end], i, que, "multi" )))
            for pro in processes:
                pro.start()
            for pro in processes:
                temp = que.get()
                #print "LENGTH OF PREDICTIONS: ", len(temp[1:])
                #print "TYPE OF PREDICTIONS: ", type(temp[1:])
                predictions_dict[temp[0]] = temp[1:]
            for pro in processes:
                pro.join()
            for i in range(len(processes)):
                predictions_list.extend(predictions_dict[i])
        else:
            classify(self.corcoeff, self.vocab, test_docs, 0, que, "multi")
            predictions_list.extend(que.get()[1:])
        #print(len(predictions_list))
        return numpy.array(predictions_list)




        """
        def predict_multilabel(self, test_docs):

            predictions_list = []
            predictions_dict = {}
            que = Queue()

            if thread_count < len(test_docs):
                div = (len(test_docs)/thread_count)+1
                processes = []
                for i in range(thread_count):
                    end = (i+1)*div
                    if end > len(test_docs):
                        end = len(test_docs)
                    processes.append(Process(target=classify, args=( self.corcoeff, self.vocab, test_docs[i*div:end], i, que, "multi" )))
                for pro in processes:
                    pro.start()
                for pro in processes:
                    temp = que.get()
                    #print "LENGTH OF PREDICTIONS: ", len(temp[1:])
                    #print "TYPE OF PREDICTIONS: ", type(temp[1:])
                    predictions_dict[temp[0]] = temp[1:]
                for pro in processes:
                    pro.join()
                for i in range(len(processes)):
                    predictions_list.extend(predictions_dict[i])
            else:
                classify(self.corcoeff, self.vocab, test_docs, 0, que, "multi")
                predictions_list.extend(que.get()[1:])
            #print(len(predictions_list))
            return numpy.array(predictions_list)
            """
