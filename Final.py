from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
stop_words = stopwords.words("english")
import cooccurence_main

import operator
import itertools
import numpy
from sys import getsizeof

from sklearn.metrics import f1_score, precision_score, recall_score
import time


#Take required user inputs
weight = input("Enter weighing algorithm in QUOTES [tf or tfidf] : ")
if weight != "tfidf" and weight != "tf":
    raise ValueError ("Unrecognised option for weighing algorithm.")
cor_type = input("Enter correlation coefficient in QUOTES [P for PMI, J for Jaccard] : ")
if cor_type != "J" and cor_type != "j" and cor_type != "P" and cor_type != "p":
    raise ValueError ("Unrecognised option for correlation coefficient.")



#----------------Functions---------------

cachedStopWords = stopwords.words("english")
def tokenize(text, word_length = 3):
  min_length = word_length
  words = map(lambda word: word.lower(), word_tokenize(text))
  words = [word for word in words if word not in cachedStopWords]
  tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
  p = re.compile('[a-zA-Z]+');
  filtered_tokens = list(filter (lambda token: p.match(token) and
                               len(token) >= min_length, tokens))
  return filtered_tokens


def get_TF(vectorizer_tf, vectorised_train_documents_tf, doc_list, raw_tf=False):
    #vocab contains term-index pair
    vocab = vectorizer_tf.vocabulary_
    #sorted_vocab contains list of terms sorted based on index
    sorted_vocab = [item[0] for item in sorted(vocab.items(), key=operator.itemgetter(1))]
    tf_values = numpy.array(vectorised_train_documents_tf[doc_list, :].sum(axis=0))[0].tolist()
    if raw_tf == False:
        #Divide by total freq
        tot = float(sum(tf_values))
        vocab_tf_array = numpy.array(tf_values)/tot
        tf_values = vocab_tf_array.tolist()
    #vocab_tf_new is a dictionary that stores tfidf sum over all docs for each term
    vocab_tf_new = dict(itertools.izip(sorted_vocab, tf_values))
    #print vocab_tf_new[sorted_vocab[-2]]
    return vocab_tf_new



def get_IDF(vectorizer_tfidf):
    idf = vectorizer_tfidf.idf_
    #print dict(zip(vectorizer_tfidf.get_feature_names(), idf))
    return dict(zip(vectorizer_tfidf.get_feature_names(), idf))



def get_TFIDF(vectorizer_tfidf, vectorised_train_documents_tfidf, doc_list):
    #vocab contains term-index pair
    vocab = vectorizer_tfidf.vocabulary_
    #sorted_vocab contains list of terms sorted based on index
    sorted_vocab = [item[0] for item in sorted(vocab.items(), key=operator.itemgetter(1))]
    #vocab_tfidf_new is a dictionary that stores tfidf sum over all docs for each term
    vocab_tfidf_new = dict(itertools.izip(sorted_vocab, numpy.array(vectorised_train_documents_tfidf[doc_list, :].sum(axis=0))[0].tolist()))
    #print vocab_tfidf_new[sorted_vocab[-2]]
    return vocab_tfidf_new




#----------------Feature Extraction--------------------------

start_time = time.time()
program_start = start_time

# List of documents & ids
documents = reuters.fileids()
#print "Documents : ", getsizeof(documents)

train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

# Transform for multilabel compatibility
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])
#print numpy.nonzero(train_labels[4, :])[0].tolist()

print "Building file list complete and it took : ", time.time() - start_time, "seconds"
start_time = time.time()


#Process all documents
if (weight == "tf" and cor_type == "J") or cor_type == "P":
    # Learn and transform documents [tf]
    vectorizer_tf = CountVectorizer(stop_words=stop_words, tokenizer=tokenize)
    vectorised_train_documents_tf = vectorizer_tf.fit_transform(train_docs)
    vectorised_test_documents_tf = vectorizer_tf.transform(test_docs)
if weight == "tfidf":
    # Learn and transform documents [tfidf]
    vectorizer_tfidf = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
    vectorised_train_documents_tfidf = vectorizer_tfidf.fit_transform(train_docs)
    vectorised_test_documents_tfidf = vectorizer_tfidf.transform(test_docs)
"""
Rows: Docs ; Columns: Terms
print vectorised_test_documents_tfidf[[1, 3], :].shape
print vectorised_test_documents_tfidf.shape
print len(train_docs), " : ", vectorised_train_documents_tfidf.shape
print len(test_docs), " : ", vectorised_test_documents_tfidf.shape
"""

#Devide features by class
if (weight == "tf" and cor_type == "J") or cor_type == "P":
    vocab_tf = {}
    for i in range(train_labels.shape[1]):
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        vocab_tf[i] = get_TF(vectorizer_tf, vectorised_train_documents_tf, classdoc_ids)
    vocab_choice = vocab_tf
if weight == "tfidf":
    vocab_tfidf = {}
    for i in range(train_labels.shape[1]):
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        vocab_tfidf[i] = get_TFIDF(vectorizer_tfidf, vectorised_train_documents_tfidf, classdoc_ids)
    vocab_choice = vocab_tfidf
"""
print numpy.nonzero(vocab_tf[4].values())
print vocab_tf[4].keys()[9], " : ", vocab_tf[4].values()[9]
"""

print "Generating term-weights complete and it took : ", time.time() - start_time, "seconds"
start_time = time.time()

#Find cooccurences for all classes
cooccurences_by_class = cooccurence_main.get_cooccurences(train_labels, train_docs)
"""
print "cooccurences_by_class : ", getsizeof(cooccurences_by_class)
print cooccurences_by_class[4].values()
"""


if cor_type == "J" or cor_type == "j":
    vocab_raw = {}
    for i in range(train_labels.shape[1]):
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        vocab_raw[i] = get_TF(vectorizer_tf, vectorised_train_documents_tf, classdoc_ids, raw_tf=True)
    corcoeff = cooccurence_main.cacl_corcoff(cooccurences_by_class, vocab_raw, cor_type)
elif cor_type == "P" or cor_type == "p":
    cooccurence_main.cacl_corcoff(cooccurences_by_class, vocab_tf, cor_type)
    corcoeff = cooccurences_by_class
#print cooccurences_by_class[4].values()


print "Generating correlation-coefficients complete and it took : ", time.time() - start_time, "seconds"
start_time = time.time()




#----------------Classification--------------------------

"""
"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# Classifier
#n_jobs, specifies the nuber of cores used, -1 is for all cores, -2 for all except 1 and so on
classifier = OneVsRestClassifier(LinearSVC(random_state=42),  n_jobs=-2)
classifier.fit(vectorised_train_documents_tfidf, train_labels)


print "The Classifier is trained and it took"
print time.time() - start_time, "seconds"
start_time = time.time()

predictions = classifier.predict(vectorised_test_documents_tfidf)

print "The Classification is complete and it took"
print time.time() - start_time, "seconds"
start_time = time.time()

"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
"""


"""
#-----------------Evaluation ----------------------
precision = precision_score(test_labels, predictions,
                            average='micro')
recall = recall_score(test_labels, predictions,
                      average='micro')
f1 = f1_score(test_labels, predictions, average='micro')

print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
        .format(precision, recall, f1))

precision = precision_score(test_labels, predictions,
                            average='macro')
recall = recall_score(test_labels, predictions,
                      average='macro')
f1 = f1_score(test_labels, predictions, average='macro')

print("Macro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
        .format(precision, recall, f1))
"""

print "Evaluation complete and it took : ", time.time() - start_time, "seconds"

print "Total time taken : ", time.time() - program_start, "seconds"
