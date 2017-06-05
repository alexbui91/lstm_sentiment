import os
import utils
import shutil
import random

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
    for sent, target in sents:
        data_y.append(target)
        docs_idx = utils.make_sentence_idx(vocabs, sent, maxlen)
        data_x.append(docs_idx)
    return data_x, data_y