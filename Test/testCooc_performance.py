from nltk.corpus import stopwords
import string

import glob
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import itertools
import time
import csv


#Retturn clean sentence
stop = stopwords.words('english') + list(string.punctuation)
def clean_text(sent):
    word_list = [word.lower() for word in word_tokenize(sent) if word.lower() not in stop]
    return word_list



files = glob.glob("/home/sounak/work/Datasets/Reuters21578-Apte-top10/training/earn/*")
all_pairs = {}
all_pairs2 = []
writeFile1 = csv.writer(open('avgSize.csv', 'w'))
writeFile2 = open('maxSize.txt', 'w')

strtTime = time.time()
# iterate over the list getting each file 
for fle in files:
    # open the file and then call .read() to get the text \
    with open(fle) as f:
        text = f.read()
        sent_tokenize_list = sent_tokenize(text)
        for sent in sent_tokenize_list:
            words = clean_text(sent)
            word_pairs = list(itertools.combinations(words, 2))
            all_pairs2.extend(word_pairs)
            for pair in word_pairs:
                if pair in all_pairs:
                    all_pairs[pair] += 1
                else:
                    all_pairs[pair] = 1
print "Process time : "
print time.time() - strtTime
print "Dict Size : {}".format(len(all_pairs))
print "List Size : {}".format(len(all_pairs2))
print "List Size [Without duplicates] : {}".format(len(set(all_pairs2)))
strtTime = time.time()

for key, val in all_pairs.items():
    writeFile1.writerow([key, val])
for pair in all_pairs2:
  print>>writeFile2, pair
print "Write time : "
print time.time() - strtTime
