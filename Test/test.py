import operator
import numpy

#frozenset, order matters?
myset1 = frozenset({"this", "is", "test"})
myset2 = frozenset({"this", "is", "test"})
myset3 = frozenset({"this", "is", "a"})
dict_test = {myset1 : "found", myset3 : "not"}
print dict_test[myset2]
print "##########################################################################################"

#keeping dict on both sides of '='
dict_test2 = {"tt" : 7}
dict_test2["tt"] = dict_test2["tt"]/2.0
print dict_test2["tt"]
print "##########################################################################################"

#test keys and values mainain same order
test_dict = {"This" : 2, "is" : 4, "a" : 3, "test" : 3, "for" : 1, "extrcting" : 1, "sorted" : 2, "lists" : 5}
print test_dict.keys()
print test_dict.values()
print [item[0] for item in sorted(test_dict.items(), key=operator.itemgetter(1))]
print "##########################################################################################"

#test order maintained during conversion
test_list = [2,4,1,5,7,6,3,8,6]
test_array = numpy.array(test_list)
print test_array
print test_array.tolist()
print "##########################################################################################"
