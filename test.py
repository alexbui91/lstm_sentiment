import os
import utils
import properties
import argparse

from model import Model
from lstm_cnn import LSTM_CNN

word_vectors, vocabs = None, None

def exe(word_vectors_file, vector_preloaded_path, test_path, sent, hidden_sizes, maxlen, mix):
    global word_vectors, vocabs
    if not maxlen:
        maxlen = properties.maxlen
    if word_vectors is None or vocabs is None:
        word_vectors, vocabs = utils.loadWordVectors(word_vectors_file, vector_preloaded_path)
    if sent:
        if sent: 
            test_x = utils.make_sentence_idx(vocabs, sent.lower(), maxlen)
            test_y = [1]
    else: 
        #auto test path_file
        test_x, test_y = utils.load_file(test_path)
    if mix is 'Y':
        combined = LSTM_CNN(word_vectors, hidden_sizes=hidden_sizes)
        errors = combined.build_test_model((test_x, test_y, maxlen))
    else:
        lstm = Model(word_vectors, hidden_sizes=hidden_sizes)
        errors = lstm.build_test_model((test_x, test_y, maxlen))
    if pred:
        print "sentiment is positive"
    else: 
        print "sentiment is negative"
    if errors:
        print("Error of test is: %.2f" % errors)
        
    

#python main.py --train='../data/50d.training_twitter_full.txt' --dev='../data/50d.dev_twitter_small.txt' --test='../data/50d.test_twitter.txt' --vectors='../data/glove.6B.50d.txt' --plvec='../data'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running LSTM')
    parser.add_argument('--vectors', type=str, default='/home/alex/Documents/nlp/data/glove.6B.50d.txt')
    parser.add_argument('--plvec', type=str, default='/home/alex/Documents/nlp/data')
    parser.add_argument('--test', type=str, default='')
    parser.add_argument('--sent', type=str, default='')
    parser.add_argument('--mix', type=str, default='Y')
    parser.add_argument('--max', type=int, default=140)
    parser.add_argument('--hs', type=list, default=[50, 2])

    args = parser.parse_args()

    exe(args.vectors, args.plvec, args.test, args.sent, args.hs, args.max, args.mix)