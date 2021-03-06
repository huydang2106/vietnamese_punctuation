import codecs
import ujson
import re
import unicodedata
import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from tqdm import tqdm
import fasttext.util
import pickle
import random
PAD = "<PAD>"
UNK = "<UNK>"
NUM = "<NUM>"
END = "</S>"


label_dict = {'O': 0,'PERIOD':1, 'COMMA':2, 'COLON':3, 'QMARK':4}
PUNCTUATION_SYM = ['','.',',',':','?']

EOS_TOKENS = ['PERIOD','QMARK']

def write_json(filename, dataset):
    with codecs.open(filename, mode="w", encoding="utf-8") as f:
        ujson.dump(dataset, f)


def word_convert(word, keep_number=True, lowercase=True):
    # convert french characters to latin equivalents
    if not keep_number:
        if is_digit(word):
            word = NUM
    if lowercase:
        word = word.lower()
    return word


def is_digit(word):
    try:
        float(word)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(word)
        return True
    except (TypeError, ValueError):
        pass
    result = re.compile(r'^[-+]?[0-9]+,[0-9]+$').match(word)
    if result:
        return True
    return False

## Preprocessing data
## List of {words: [], labels: []} - each dict present a sequence
def load_dataset(filename, keep_number=False, lowercase=True, max_len_seq = 256):
    def remove_list(ls,token):
        while token in ls:
            ls.remove(token)
        return ls
    dataset = []
    with codecs.open(filename, mode="r", encoding="utf-8") as f:
        words, labels = [], []
        random_seq_length = random.randint(30,256)
        for line in tqdm(f):
            line = line.lstrip().rstrip()
            line = line.split(" ")
            if len(line) != 2:
                # means read whole one sentence
                print('skip for len(line) != 2')
                print(line)
                continue
            else:
                word = line[0]
                label = line[1]
                word = word_convert(word, keep_number=keep_number, lowercase=lowercase)
                if len(words) < random_seq_length:
                    # print('append')
                    words.append(word)
                    labels.append(label)
                if len(words) > random_seq_length  and len(words) <= max_len_seq:
                    # print('> random seqence length',random_seq_length)
                    # print(len(words),len(labels))
                    while len(words) < max_len_seq:
                        words.append(PAD)
                        labels.append("0")
                if len(words) > max_len_seq:
                    print('>max_len_seq')
                    print(words)
                    input()
                    break
                if len(words) >= random_seq_length:
                    # print(' '.join(words))
                    # input()
                    while len(words) < max_len_seq:
                        words.append(PAD)
                        labels.append("0")
                    dataset.append({"words": words, "labels": labels})
                    random_seq_length = random.randint(30,256)
                    i = len(words) - 1
    #                print(len(tags))
                    while i > 0:
                        if labels[i] in EOS_TOKENS:
#                            print(i)
                            if i == (len(words) - 1):
                                words, labels = [], []
                            else:
                                w = words[i + 1:]
                                
                                l = labels[i + 1:]
                                w = remove_list(w,PAD)
                                l = remove_list(l,"0")
                                words, labels = [], []
                                words = w
                                labels = l
                            break
                        else:
                            i = i - 1
                    if len(words) == max_len_seq:
                        words, labels = [], []
        dataset.append({"words": words, "labels": labels})
    f.close()


    return dataset

def load_word_embedding(embedding_file):
    print('Loading word embeddings...')
    embeddings_index = {}
    # f = codecs.open(embedding_file, encoding='utf-8')
    # for line in f:
    #     values = line.rstrip().rsplit(' ')
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()
   
    # fasttext.util.download_model('vi', if_exists='ignore')  # English
    ft = fasttext.load_model(embedding_file)
    
    # data = {}
    
    for w in tqdm(ft.words):
        embeddings_index[w] = np.array(ft.get_word_vector(w),dtype='float32')
    return embeddings_index
    
def build_word_vocab(datasets):
    vocab_size = 50000
    word_counter = Counter()
    for dataset in tqdm(datasets):
        for record in dataset:
            words = record["words"]
            for word in words:
                word_counter[word] += 1
    word_vocab = [PAD, UNK, NUM] + [word for word, _ in word_counter.most_common(vocab_size) if word != NUM]
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict, word_vocab

def build_char_vocab(datasets):
    char_counter = Counter()
    for dataset in tqdm(datasets):
        for record in dataset:
            for word in record["words"]:
                for char in word:
                    char_counter[char] += 1
    char_vocab = [PAD, UNK] + [char for char, _ in char_counter.most_common()]
    char_dict = dict([(word, idx) for idx, word in enumerate(char_vocab)])
    return char_dict
'''
build vocab 
'''
def build_word_vocab_pretrained(datasets):
    # [{words: list of words, labels: list of labels}]
    word_counter = Counter()
    for dataset in tqdm(datasets):
        for record in dataset:
            words = record["words"]
            for word in words:
                word_counter[word] += 1
    # build word dict
    word_vocab = [word for word, _ in word_counter.most_common() if word != NUM]
    word_vocab = [PAD, UNK, NUM] + list(set(word_vocab))
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict

'''
get all embedding from pretrained. if one word is not in pretrained dictionary (embedding_index),
random the embedding vector
'''
def filter_glove_emb(word_dict, embedding_index):
    
    # filter embeddings
    dim = 300
    scale = np.sqrt(3.0 / dim)
    vectors = np.random.uniform(-scale, scale, [len(word_dict), dim])
    
    for word in word_dict.keys():
        embedding_vector = embedding_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            index = list(word_dict.keys()).index(word)
            vectors[index] = embedding_vector
    return vectors

def build_dataset(data, word_dict, char_dict, label_dict):
    dataset = []
    for record in tqdm(data):
        chars_list = []
        words = []
        for word in record["words"]:
            chars = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            chars_list.append(chars)
            word = word_convert(word, keep_number=False, lowercase=True)
            words.append(word_dict[word] if word in word_dict else word_dict[UNK])
        labels = [label_dict[label] for label in record["labels"]]
        dataset.append({"words": words, "chars": chars_list, "labels": labels})
    return dataset


def process_data(config):
    print('process data')
    print('config')
    print(config)
    train_data = load_dataset(os.path.join(config["raw_path"], "train.txt"))
    dev_data = load_dataset(os.path.join(config["raw_path"], "valid.txt"))
    test_data = load_dataset(os.path.join(config["raw_path"], "test.txt"))
    embedding_file = os.path.join(config["Word2vec_path"], "cc.vi.300.vec")

    embedding_index = load_word_embedding(embedding_file)
    
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    # build vocabulary
    word_dict = build_word_vocab_pretrained([train_data, dev_data, test_data])
    vectors = filter_glove_emb(word_dict, embedding_index)
    np.savez_compressed(config["word_embedding"], embeddings=vectors)
# build char dict
    # rebuild dataset, keep number
    train_data = load_dataset(os.path.join(config["raw_path"], "train.txt"), keep_number=True,
                              lowercase=config["char_lowercase"])
    dev_data = load_dataset(os.path.join(config["raw_path"], "valid.txt"), keep_number=True,
                            lowercase=config["char_lowercase"])
    test_data = load_dataset(os.path.join(config["raw_path"], "test.txt"), keep_number=True,
                             lowercase=config["char_lowercase"])
    char_dict = build_char_vocab([train_data, dev_data, test_data])
    # create indices dataset
    train_set = build_dataset(train_data, word_dict, char_dict, label_dict)
    dev_set = build_dataset(dev_data, word_dict, char_dict, label_dict)
    test_set = build_dataset(test_data, word_dict, char_dict, label_dict)
    print('Train set:', len(train_set))
    print('Dev set:', len(dev_set))
    vocab = {"word_dict": word_dict, "char_dict": char_dict, "label_dict": label_dict}
    # write to file
    write_json(os.path.join(config["save_path"], "vocab.json"), vocab)
    write_json(os.path.join(config["save_path"], "train.json"), train_set)
    write_json(os.path.join(config["save_path"], "dev.json"), dev_set)
    write_json(os.path.join(config["save_path"], "test.json"), test_set)
    return label_dict