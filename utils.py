import theano
import theano.tensor as T
import pickle
import numpy as np
import properties
import os.path as path
from data import Data


def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)


def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)


def Tanh(x):
    y = T.tanh(x)
    return(y)


def Iden(x):
    y = x
    return(y)


def dropout_from_layer(rng, layer, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output


def find_largest_number(num1, num2, num3):
    largest = num1
    if (num1 >= num2) and (num1 >= num3):
       largest = num1
    elif (num2 >= num1) and (num2 >= num3):
       largest = num2
    else:
       largest = num3
    return largest


def save_file(name, obj):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_file(pathfile):
    if path.exists(pathfile):
        with open(pathfile, 'rb') as f:
            data = pickle.load(f)
        return data 


def loadWordVectors(file, data_path):
    d = Data(file)
    d.loadWordVectorsFromText(data_path)
    return d.vectors, d.vocabs


def make_sentence_idx(vocabs, sent, max_sent_length):
    sent_v = list()
    sent_length = len(sent)
    for i in xrange(max_sent_length):
        if i < sent_length:
            if sent[i] in vocabs:
                sent_v.append(vocabs[sent[i]])
            else: 
                sent_v.append(0)
        else:
            sent_v.append(0)
    return sent_v


def get_num_words(vocabs, sent):
    length = 0
    words = sent.split()
    for word in words:
        if word in vocabs:
            length += 1
    return length


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def check_array_full_zeros(arr):
    for x in arr:
        if not x:
            return False
    return True

def save_layer_params(self, layers, name):
    # now = time.time()
    params = [param.get_value() for param in layers.params]
    utils.save_file('%s.txt' % name, params)