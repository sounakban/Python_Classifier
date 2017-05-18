import cooccurence_extract

new = {tuple(k):v for k,v in cooccurence_extract.process_text("This is a proper working test. To see whether the proper test algo is working correctlt").items()}
for k in new.keys():
    print k[0] == k[1], len(k)

print "##########################################################################################"

test_min = [0, 3 , 2, 4, 7, 2, 0, 4]
print test_min.index(min((test_min)))
