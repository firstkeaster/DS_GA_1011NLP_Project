from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl


"""Data preprocessing"""

def zh_en_reader(dataset, filepath):
    with open(filepath, encoding='utf-8') as f:
        # f.readline()
        lines = f.readlines()
        for line in lines:
            #dataset.append([x.strip('. \n') for x in line.split('\t')])
            dataset.append(line)
        f.close()

## preprocessor: remove multiple spaces
def data_preprocessor(dataset):
    for i, _ in enumerate(dataset):
        dataset[i] = ' '.join([word for word in dataset[i].strip('\n').split(' ') if word != ''])
    return

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<pad>", 1: "<unk>", 2: "SOS", 3: "EOS"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word.lower())

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def prepareData_zh_en_ensemble(lang1, lang2, lang1_data, lang2_data):
    input_lang, output_lang = Lang(lang1), Lang(lang2)
    print("Counting words...")
    pairs = []
    for i,_ in enumerate(lang1_data):
        input_lang.addSentence(lang1_data[i])
        output_lang.addSentence(lang2_data[i])
        pairs.append([lang1_data[i], lang2_data[i]])
    ## add <unk> to dictionary
    ## input_lang.addSentence('<unk> <unk>')
    ## output_lang.addSentence('<unk> <unk>')
    ## finished adding unk
    print("Read %s sentence pairs" % len(pairs))
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence, UNK_token):
    s =  [lang.word2index.get(word.lower(), UNK_token) for word in sentence.split(' ')]
    return s

def ConvertSentence2numpy(input_lang, output_lang, pairs, UNK_token, EOS_token, MAX_LEN = None):
    input_sentences = []
    output_sentences = []
    if MAX_LEN == None:
        for i in range(len(pairs)):
            s1 = indexesFromSentence(input_lang, pairs[i][0], UNK_token)
            s1.append(EOS_token)
            input_sentences.append(np.array(s1))
            s2 = indexesFromSentence(output_lang, pairs[i][1], UNK_token)
            s2.append(EOS_token)
            output_sentences.append(np.array(s2))
    else:
          for i in range(len(pairs)):
            s1 = indexesFromSentence(input_lang, pairs[i][0], UNK_token)
            if len(s1)<MAX_LEN:
                s1.append(EOS_token)
                input_sentences.append(np.array(s1))
                s2 = indexesFromSentence(output_lang, pairs[i][1], UNK_token)
                s2.append(EOS_token)
                output_sentences.append(np.array(s2))
    return input_sentences, output_sentences

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points, title='trainCurve'):
    fig, ax = plt.subplots()
    lines = {}
    lines['cur'], = ax.plot(range(len(points)), points, label='train loss')
    ax.set_title(title)

def Load_pretrained_emb(n, lang, emb_dataDir):
    words_to_load = n

    if lang.name == 'zh':
        vec = emb_dataDir + '//wiki.zh.vec'
    elif lang.name == 'en':
        vec = emb_dataDir + '//wiki.en.vec'
    elif lang.name == 'vi':
        vec = emb_dataDir + '//wiki.vi.vec'

    with open(vec, 'r', encoding = 'utf-8') as f:
        embedding_matrix = np.zeros([words_to_load, 300])
        word2id_emb = {}
        id2word_emb = []
        next(f)
        for i, line in enumerate(f):
            if i >= words_to_load:
                break
            s = line.split()

            if len(s[1:])== 300:
                id2word_emb.append(s[0])
                word2id_emb[s[0]] = i
                embedding_matrix[i] = np.asarray(s[1:])

    print("Embedding_matrix shape is {}".format(embedding_matrix.shape))
    print("Embedding Vocabualry size is {}".format(len(id2word_emb)))

    """Build pretrained Embedding Matrix"""
    pre_emb_matrix = np.zeros([lang.n_words, 300])

    i = 0
    for word in lang.word2index.keys():
        try:
            id_pretrain = word2id_emb[word]
            pre_emb_matrix[lang.word2index[word]] = embedding_matrix[id_pretrain]
        except KeyError:
            pre_emb_matrix[lang.word2index[word]] = np.zeros(300)
            i = i+1
    print("Size of the pretrained embedding matrix for ", lang.name, " is ", pre_emb_matrix.shape)
    print("{} words appear in the pretrained dataset".format(1-i/lang.n_words))

    return pre_emb_matrix

class TransDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, input_sentences, target_sentences):
        """
        @param input_sentences: list of input sentences
        @param target_sentences: list of target sentences

        """
        self.input_sentences, self.target_sentences = input_sentences, target_sentences
        assert (len(self.input_sentences)==len(self.target_sentences))

    def __len__(self):
        return len(self.input_sentences)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        return [self.input_sentences[key], len(self.input_sentences[key]), \
                self.target_sentences[key], len(self.target_sentences[key])]

def vocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    BUFFER_NUMBER = 0

    input_list = []
    target_list = []

    input_length_list = []
    target_length_list = []

    for datum in batch:
        input_length_list.append(datum[1])
        target_length_list.append(datum[3])
    # padding
    #MAX_WORD_LENGTH
    input_max_length = max(input_length_list)
    target_max_length = max(target_length_list) + BUFFER_NUMBER
    for datum in batch:
        padded_vec_input = np.pad(np.array(datum[0]),
                                pad_width=((0, input_max_length-len(datum[0]))),
                                mode="constant", constant_values=0)
        padded_vec_target = np.pad(np.array(datum[2]),
                                pad_width=((0, target_max_length-len(datum[2]))),
                                mode="constant", constant_values=0)
        input_list.append(padded_vec_input)
        target_list.append(padded_vec_target)
    ind_dec_order = np.argsort(input_length_list)[::-1]
    input_list = np.array(input_list)[ind_dec_order]
    target_list = np.array(target_list)[ind_dec_order]
#     length_list = np.array(length_list)[ind_dec_order]
#     label_list = np.array(label_list)[ind_dec_order]
    return [torch.from_numpy(np.array(input_list)), torch.from_numpy(np.array(target_list))]#, torch.LongTensor(length_list), torch.LongTensor(label_list)]


def pkl_dumper(objct, file_name):
    with open(file_name, 'wb+') as f:
        pkl.dump(objct, f, protocol=None)

def pkl_loader(file_name):
    with open(file_name, 'rb') as f:
        objct = pkl.load(f, encoding = 'bytes')
    return(objct)
