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

test_min = [0, 3 , 2, 4, 7, 6, 9, 1]
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
