import os
import shutil
import numpy as np
import utils
import properties
import random

from model import Model


word_vectors, vocabs = None, None


def mix_data():    
    train_path = "data/train"
    test_path = "data/test"
    final_path = "data/all"
    create_path_if_not_exist(final_path)
    copy_all(train_path + "/pos", final_path + "/pos", "tr")
    copy_all(train_path + "/neg", final_path + "/neg", "tr")
    copy_all(test_path + "/pos", final_path + "/pos", "ts")
    copy_all(test_path + "/neg", final_path + "/neg", "ts")


def separate_data(directory, dev, test): 
    train_path = directory + '/train_m'
    dev_path = directory + '/dev_m'
    test_path = directory + '/test_m'
    create_path_if_not_exist(train_path + "/pos")
    create_path_if_not_exist(dev_path + "/pos")
    create_path_if_not_exist(test_path + "/pos")
    create_path_if_not_exist(train_path + "/neg")
    create_path_if_not_exist(dev_path + "/neg")
    create_path_if_not_exist(test_path + "/neg")
    if not os.path.exists(directory):
        print("Source file is not existed")
        return
    pos_files = os.listdir(directory + "/all/pos")
    neg_files = os.listdir(directory + "/all/neg")
    pos_files_length = len(pos_files)
    neg_files_length = len(neg_files)
    if (dev + test) >= (pos_files_length + neg_files_length):
        print("Dataset length is overloading")
        return
    dev_pos = dev / 2
    test_pos = test / 2
    dev_test_pos = dev_pos + test_pos
    train_pos = pos_files_length - dev_test_pos
    train_neg = neg_files_length - dev_test_pos
    np.random.shuffle(pos_files)
    np.random.shuffle(neg_files)
    copy_list_to_dir(directory, train_path, dev_path, test_path, dev_pos, dev_test_pos, pos_files, pos_files_length, "pos")
    copy_list_to_dir(directory, train_path, dev_path, test_path, dev_pos, dev_test_pos, neg_files, neg_files_length, "neg")


def shuffle_separate(vocabs, directory, dev, test):
    if not os.path.exists(directory + "/all"):
        print("Source file is not existed")
        return
    sents_pos, length_pos = load_data_in_classified_folder(directory + "/all/pos", vocabs)
    sents_neg, length_neg = load_data_in_classified_folder(directory + "/all/pos", vocabs)
    # sents_pos = sents_pos[0:(len(sents_neg)/2)]
    sents = zip(sents_pos, np.ones(len(sents_pos))) + zip(sents_neg, np.zeros(len(sents_neg)))
    random.shuffle(sents)
    start_test = dev + test
    dev_set = sents[:dev]
    test_set = sents[dev:start_test]
    train_set = sents[start_test:]
    maxlen = np.max([length_pos, length_neg])
    return train_set, dev_set, test_set, maxlen


def copy_list_to_dir(directory, train_path, dev_path, test_path, dev_end, test_end, files, length, subfolder):
    i = 0
    while i < length:
        if i < dev_end:
            shutil.copy2(os.path.join(directory + "/all/" + subfolder, files[i]), os.path.join(dev_path + "/" + subfolder, files[i]))
        elif i < test_end:
            shutil.copy2(os.path.join(directory + "/all/" + subfolder, files[i]), os.path.join(test_path + "/" + subfolder, files[i]))
        else:
            shutil.copy2(os.path.join(directory + "/all/" + subfolder, files[i]), os.path.join(train_path + "/" + subfolder, files[i]))
        i += 1


def create_path_if_not_exist(name):
    if not os.path.exists(name):
        os.makedirs(name)


def copy_all(from_dir, to_dir, prefix):
    if not os.path.exists(from_dir):
        print("Source file is not existed")
        return
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    for filename in os.listdir(from_dir):
        shutil.copy2(os.path.join(from_dir, filename),os.path.join(to_dir, prefix + filename))


def load_data_in_directory(directory, vocabs):
    p_sents, p_maxlen = load_data_in_classified_folder(directory + "/pos", vocabs)
    n_sents, n_maxlen = load_data_in_classified_folder(directory + "/neg", vocabs)
    if p_maxlen < n_maxlen:
        p_maxlen = n_maxlen
    return p_sents, n_sents, p_maxlen
   

def load_data_in_classified_folder(path, vocabs):
    maxlen = 0
    sents = list()
    for filename in os.listdir(path):
        if filename.endswith(".txt"): 
            with open(os.path.join(path, filename), 'rb') as f:
                docs = f.read()
                docs = docs.lower()
                sents.append(docs)
                length = utils.get_num_words(vocabs, docs)
                if maxlen < length:
                    maxlen = length
    return sents, maxlen


# def process_data(vocabs, sents, target, maxlen):
#     data_x, data_y = list(), list()
#     for sent in sents:
#         data_y.append(target)
#         docs_idx = utils.make_sentence_idx(vocabs, sent, maxlen)
#         data_x.append(docs_idx)
#     return data_x, data_y


def process_data(vocabs, sents, maxlen):
    data_x, data_y = list(), list()
    print(len(sents))
    for sent, target in sents:
        data_y.append(target)
        docs_idx = utils.make_sentence_idx(vocabs, sent, maxlen)
        data_x.append(docs_idx)
    return data_x, data_y


#word_vector_path="../data/glove.6B.300d.txt"
#import main; main.exe(word_vectors_file="../data/glove.6B.50d.txt", word_vectors_path="./data")
#import main; main.exe(word_vectors_file="../data/glove_text8.txt", word_vectors_path="../cnn_sentiment/data")
def exe(word_vectors_file, word_vectors_path="data/", hidden_sizes=[50, 10, 2], is_reload_data=False):
    global word_vectors, vocabs
    data_path = "data"
    train_path = data_path + "/train"
    dev_path = data_path + "/dev"
    test_path = data_path + "/test"
    datafile = data_path + "/dataset.txt"
    if word_vectors is None or vocabs is None:
        word_vectors, vocabs = utils.loadWordVectors(word_vectors_file, word_vectors_path)
    if os.path.exists(datafile) and not is_reload_data:
        dataset = utils.load_file(datafile)
        dataset['maxlen'] = properties.maxlen
    else: 
        dataset = dict()
        # train_pos_sents, train_neg_sents, train_len = load_data_in_directory(train_path, vocabs)
        # dev_pos_sents, dev_neg_sents, dev_len = load_data_in_directory(dev_path, vocabs)
        # test_pos_sents, test_neg_sents, test_len = load_data_in_directory(test_path, vocabs)
        # maxlen = utils.find_largest_number(train_len, dev_len, test_len)
        # maxlen = properties.maxlen
        # train_pos_x, train_pos_y = process_data(vocabs, train_pos_sents, 1, maxlen)
        # train_neg_x, train_neg_y = process_data(vocabs, train_neg_sents, 0, maxlen)
        # dev_pos_x, dev_pos_y = process_data(vocabs, dev_pos_sents, 1, maxlen)
        # dev_neg_x, dev_neg_y = process_data(vocabs, dev_neg_sents, 0, maxlen)
        # test_pos_x, test_pos_y = process_data(vocabs, test_pos_sents, 1, maxlen)
        # test_neg_x, test_neg_y = process_data(vocabs, test_neg_sents, 0, maxlen)

        train_set, dev_set, test_set, maxlen = shuffle_separate(vocabs, data_path, properties.dev_size, properties.test_size)
        maxlen = properties.maxlen
        train_x, train_y = process_data(vocabs, train_set, maxlen)
        dev_x, dev_y = process_data(vocabs, dev_set, maxlen)
        test_x, test_y = process_data(vocabs, test_set, maxlen)
        dataset['train'] = (train_x, train_y)
        dataset['dev'] = (dev_x, dev_y)
        dataset['test'] = (test_x, test_y)
        dataset['maxlen'] = maxlen
        utils.save_file(datafile, dataset)
    model = Model(word_vectors, hidden_sizes=hidden_sizes)
    model.train(dataset['train'], dataset['dev'], dataset['test'], dataset['maxlen'])