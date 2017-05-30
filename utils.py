import theano
import theano.tensor as T
import pickle
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