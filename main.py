import os
import shutil
import numpy as np
import utils


words_vector, vocabs = None, None


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


def load_data_in_directory(directory):
    pos_x, pos_y = load_data_in_classified_folder(directory + "/pos", 1)
    neg_x, neg_y = load_data_in_classified_folder(directory + "/neg", 0)
    return np.concatenate((pos_x, neg_x)), np.concatenate((pos_y, neg_y))
   


def load_data_in_classified_folder(path, target):
    data_x, data_y = list(), list()
    for filename in os.listdir(path):
        if filename.endswith(".txt"): 
            with open(os.path.join(path, filename), 'rb') as f:
                data_y.append(target)
                docs = f.read()
                docs_idx = make_sentence_idx(docs)
                data_x.append(docs_idx)
    return data_x, data_y


def make_sentence_idx(docs):
    global vocabs
    results_x = list()
    docs = docs.lower()
    words = docs.split(" ")
    for word in words:
        if word in vocabs:
            results_x.append(vocabs[word])
    return results_x

#word_vector_path="../data/glove.6B.300d.txt"
def exe(word_vectors_path, is_reload_data=False):
    data_path = "data"
    train_path = data_path + "/train_m"
    dev_path = data_path + "/dev_m"
    test_path = data_path + "/test_m"
    datafile = data_path + "/dataset.txt"
    if word_vectors is None or vocabs is None:
        word_vectors, vocabs = utils.loadWordVectors(word_vectors_path)
    if os.path.exists(datafile):
    if os.path.exists(datafile) and not is_reload_data:
        with open(datafile, 'rb') as f:
            dataset = pickle.load(f)
            train_x, train_y = dataset['train']
            dev_x, dev_y = dataset['dev']
            test_x, test_y = dataset['test']
    else: 
        dataset = dict()
        train_x, train_y = load_data_in_directory(train_path)
        dev_x, dev_y = load_data_in_directory(dev_path)
        test_x, test_y = load_data_in_directory(train_path)
        dataset['train'] = (train_x, train_y)
        dataset['dev'] = (dev_x, dev_y)
        dataset['test'] = (test_x, test_y)
        utils.save_file(datafile, dataset)