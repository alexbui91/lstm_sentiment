import os
import numpy as np
import utils
import properties
import argparse

from model import Model
from lstm_cnn import LSTM_CNN

word_vectors, vocabs = None, None

#word_vector_path="../data/glove.6B.300d.txt"
#import main; main.exe(word_vectors_file="../data/glove.6B.50d.txt", word_vectors_path="./data")
#import main; main.exe(word_vectors_file="../data/glove_text8.txt", word_vectors_path="../cnn_sentiment/data")
def exe(word_vectors_file, vector_preloaded_path, train_path, dev_path, test_path, hidden_sizes, maxlen):
    global word_vectors, vocabs
    if os.path.exists(train_path) and os.path.exists(dev_path) and os.path.exists(test_path):
        train = utils.load_file(train_path)
        dev = utils.load_file(dev_path)
        test = utils.load_file(test_path)
    else: 
        raise NotImplementedError()
    if word_vectors is None or vocabs is None:
        word_vectors, vocabs = utils.loadWordVectors(word_vectors_file, vector_preloaded_path)
    if not maxlen:
        maxlen = properties.maxlen
    # model = Model(word_vectors, hidden_sizes=hidden_sizes)
    model = LSTM_CNN(word_vectors, hidden_sizes=hidden_sizes)
    model.train(train, dev, test, maxlen)

#python main.py --train='../data/50d.training_twitter_full.txt' --dev='../data/50d.dev_twitter_small.txt' --test='../data/50d.test_twitter.txt' --vectors='../data/glove.6B.50d.txt' --plvec='../data'


parser = argparse.ArgumentParser(description='Running LSTM')
parser.add_argument('--vectors', type=str, default='/home/alex/Documents/nlp/data/glove.6B.50d.txt')
parser.add_argument('--plvec', type=str, default='/home/alex/Documents/nlp/data')
parser.add_argument('--train', type=str, default='/home/alex/Documents/nlp/code/data/50d.training_twitter_small.txt')
parser.add_argument('--dev', type=str, default='/home/alex/Documents/nlp/code/data/50d.dev_twitter_small.txt')
parser.add_argument('--test', type=str, default='/home/alex/Documents/nlp/code/data/50d.test_twitter.txt')
parser.add_argument('--max', type=int, default=140)
parser.add_argument('--hs', type=list, default=[50, 10, 2])

args = parser.parse_args()

exe(args.vectors, args.plvec, args.train, args.dev, args.test, args.hs, args.max)