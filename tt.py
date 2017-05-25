#Feature Extraction
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
import cooccurence_main
from text_processing import tokenize, get_TF, get_TFIDF, freqToProbability
import numpy
from cooccurence_extract import process_text
from cooccurence_main import add_pairs
import operator, itertools
from functools import partial


#Common
from sys import getsizeof

# List documents & ids
documents = reuters.fileids()

train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]


#print train_docs[1:3]


#Process all documents
# Learn and transform documents [tf]
vectorizer_tf = CountVectorizer(stop_words=stop_words, tokenizer=tokenize)
vectorised_train_documents_tf = vectorizer_tf.fit_transform(train_docs[1:3])
vocab_tf = get_TF(vectorizer_tf, vectorised_train_documents_tf, [0,1])
vocab = vectorizer_tf.vocabulary_
sorted_vocab = [item[0] for item in sorted(vocab.items(), key=operator.itemgetter(1))]
tf_values = numpy.array(vectorised_train_documents_tf[[0,1], :].sum(axis=0))[0].tolist()
vocab_tf = dict(itertools.izip(sorted_vocab, tf_values))
vocab_tf_new = {key: value for key, value in vocab_tf.items() if value != 0}
vocab_tf = freqToProbability(vocab_tf_new)
print vocab_tf_new
print

coocs = map(process_text, train_docs[1:3])
tot_coocs = {}
map(lambda fil_cooc: map(partial(add_pairs, tot_coocs, fil_cooc), fil_cooc.keys()), coocs)
print tot_coocs



# Transform for multilabel compatibility
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])

print numpy.nonzero(train_labels[:, 3])
