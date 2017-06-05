import cooccurence_extract
from multiprocessing import Process, Queue, Pool

test_text = "This is a proper working test. To see whether the proper test algo is working correctlt"
new = {tuple(k):v for k,v in cooccurence_extract.process_text(test_text).items()}
for k in new.keys():
    print k[0] == k[1], len(k)
print new.keys()
print new.values()
print "##########################################################################################"

pool = Pool(processes=3)
file_coocs = pool.imap_unordered(cooccurence_extract.process_text, [test_text, test_text])
pool.close()
pool.join()
for cooc in file_coocs:
    print "1:", {tuple(k):v for k,v in cooc.items()}.keys()
    print "1:", cooc.values()



print "##########################################################################################"

test_min = [5, 3 , 2, 4, 7, 6, 9, 8]
test_min.append(0)
print test_min
print test_min.index(min((test_min)))
print test_min[2:9]
test_copy = list(test_min)
test_copy.sort(reverse=True)
print test_min
print test_copy
print "##########################################################################################"

def Qtest(X, queue):
    new = []
    for x in X:
        new.append(x**2)
    queue.put(new)

queue = Queue()
proc = []
proc.append( Process(target=Qtest, args=(test_min[0:3], queue)) )
proc.append( Process(target=Qtest, args=(test_min[3:6], queue)) )
proc.append( Process(target=Qtest, args=(test_min[6:9], queue)) )
for p in proc:
    p.start()
for p in proc:
    p.join()
for p in proc:
    print queue.get()
print "##########################################################################################"

from functools import partial

def foo(test_passReference, x):
    if x > 4:
        test_passReference.append(x)
        return x

test_passReference = []
test_map = map(partial(foo, test_passReference), test_min)
print "List : ", test_map
test_map = filter(None, test_map)
print "After Filter : ", test_map
print "Pass by reference : ", test_passReference
print "Sum : ", sum(test_map)
print "##########################################################################################"

from copula_utils import score2pred

test_scores = [-1123, -586, -754, -1098, -645, -478]
print test_scores
print score2pred(test_scores, "multi")
print "##########################################################################################"

#Keep top n elements of a dictionary
import operator

test_dic = {'a':1, 'b':84, 'c': 3, 'd':0, 'e':43, 'f': 9, 'g':105, 'h':83, 'i': 319, }
test_dic = dict(sorted(test_dic.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
print test_dic
print "##########################################################################################"

#Test creating list through map and creating numpy matrix
import numpy

test1 = [[1,2],[3,4],[5,6],[7,8]]
test = []
test.extend(map(lambda x : x, test1))
print test
test = numpy.vstack([ test,[9,10] ])
print test
test = []
test.extend(map(lambda x : x, test_scores))
print test
x = numpy.array(test1)
print x.shape
print "##########################################################################################"

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
print X.shape
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)
print X_new.shape
print selector.get_support(indices = True)
print "##########################################################################################"

print len("This is a test string.")
