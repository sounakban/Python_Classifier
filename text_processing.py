from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re       #for regular expression

import operator
import itertools
import numpy




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
    vocab_tf = dict(itertools.izip(sorted_vocab, tf_values))
    vocab_tf_new = {key: value for key, value in vocab_tf.items() if value != 0}
    """
    #print results
    print "get_TF 1 : \n", [vocab_tf[key] for key in vocab_tf.keys()[:15]]
    print "get_TF 2 : \n", [vocab_tf_new[key] for key in vocab_tf_new.keys()[:15]]
    """
    return vocab_tf_new



def get_IDF(vectorizer_tfidf):
    idf = vectorizer_tfidf.idf_
    return dict(zip(vectorizer_tfidf.get_feature_names(), idf))



def get_TFIDF(vectorizer_tfidf, vectorised_train_documents_tfidf, doc_list):
    #vocab contains term-index pair
    vocab = vectorizer_tfidf.vocabulary_
    #sorted_vocab contains list of terms sorted based on index
    sorted_vocab = [item[0] for item in sorted(vocab.items(), key=operator.itemgetter(1))]
    #vocab_tfidf_new is a dictionary that stores tfidf sum over all docs for each term
    vocab_tfidf = dict(itertools.izip(sorted_vocab, numpy.array(vectorised_train_documents_tfidf[doc_list, :].sum(axis=0))[0].tolist()))
    vocab_tfidf_new = {key: value for key, value in vocab_tfidf.items() if value != 0}
    #print vocab_tfidf_new[sorted_vocab[-2]]
    return vocab_tfidf_new
