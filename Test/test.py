myset1 = frozenset({"this", "is", "test"})
myset2 = frozenset({"this", "is", "test"})
myset3 = frozenset({"this", "is", "a"})
dict_test = {myset1 : "found", myset3 : "not"}
print dict_test[myset2]


dict_test2 = {"tt" : 7}
dict_test2["tt"] = dict_test2["tt"]/2.0
print dict_test2["tt"]
