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

from sacrebleu import raw_corpus_bleu
from torchnlp.metrics import get_moses_multi_bleu
from torch.optim.lr_scheduler import ExponentialLR

from utils import *
from model_new_simple import *
from evaluation_fs import *

BATCH_SIZE = 64
EMB_SIZE = 300
HIDDEN_SIZE = 384
EPOCH = 15

teacher_forcing_ratio = 0.9
learning_rate = 3e-4
print_every = 50
val_every = 500

nowpath = os.path.abspath('.')
zh_en_dataDir = nowpath + '//dataset//iwsltzhen'
vi_en_dataDir = nowpath + '//dataset//iwsltvien'
emb_dataDir = nowpath + '//dataset//embedding'
res_dataDir = nowpath + '//results'
res_dataDir_cn = nowpath + '//results//zh_new_attn'

MAX_LEN_filter = 40

"""=============================================================================="""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU model:{}'.format(torch.cuda.get_device_name(0)))

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

train_en = []
zh_en_reader(train_en, zh_en_dataDir + '//train.tok.en')
train_zh = []
zh_en_reader(train_zh, zh_en_dataDir + '//train.tok.zh')
dev_zh = []
zh_en_reader(dev_zh, zh_en_dataDir + '//dev.tok.zh')
dev_en = []
zh_en_reader(dev_en, zh_en_dataDir + '//dev.tok.en')
test_zh = []
zh_en_reader(test_zh, zh_en_dataDir + '//test.tok.zh')
test_en = []
zh_en_reader(test_en, zh_en_dataDir + '//test.tok.en')

data_preprocessor(dev_zh)
data_preprocessor(dev_en)
data_preprocessor(train_en)
data_preprocessor(train_zh)
data_preprocessor(test_zh)
data_preprocessor(test_en)

input_lang, output_lang, pairs = prepareData_zh_en_ensemble('zh', 'en', train_zh, train_en)
input_train, output_train = ConvertSentence2numpy(input_lang, output_lang, pairs, UNK_token, EOS_token, MAX_LEN_filter)
input_test, output_test = ConvertSentence2numpy(input_lang, output_lang, list(zip(test_zh, test_en)), UNK_token, EOS_token)
input_dev, output_dev = ConvertSentence2numpy(input_lang, output_lang, list(zip(dev_zh, dev_en)), UNK_token, EOS_token)

print(len(input_train)/len(train_zh), ' remains after filter.')
pre_zh_emb_matrix = Load_pretrained_emb(100000, input_lang, emb_dataDir)
pre_en_emb_matrix = Load_pretrained_emb(100000, output_lang, emb_dataDir)
print(torch.cuda.memory_allocated())

train_dataset = TransDataset(input_train, output_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True)
val_dataset = TransDataset(input_dev, output_dev)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=1,
                                           collate_fn=vocab_collate_func,
                                           shuffle=False)

encoder = EncoderRNN(input_lang.n_words, EMB_SIZE, HIDDEN_SIZE, pre_zh_emb_matrix).to(device)
decoder = BahdanauAttnDecoderRNN(EMB_SIZE, HIDDEN_SIZE, output_lang.n_words, pre_en_emb_matrix, dropout_p=0.5).to(device)
print('num_parameters_encoder:', sum(p.numel() for p in encoder.parameters()))
print('num_parameters_decoder:', sum(p.numel() for p in decoder.parameters()))
print('hidSize:{}'.format(HIDDEN_SIZE))

encoder.load_state_dict(torch.load(res_dataDir_cn + "//zh_encoder_batch_1210_simple_300_384_15.pth"))
decoder.load_state_dict(torch.load(res_dataDir_cn + "//zh_attn_decoder_batch_1210_simple_300_384_15.pth"))

train_loss_record = []
val_loss_record = []
step_record = []
val_record = []

print_loss_total = 0
total_step = len(train_loader)

criterion = nn.NLLLoss()

start = time.time()
EarlyTermination = False
max_Bleu = 0
# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
# scheduler_enc = ExponentialLR(encoder_optimizer, gamma=0.1)
# scheduler_dec = ExponentialLR(decoder_optimizer, gamma=0.1)

for epoch in range(EPOCH):
    if EarlyTermination == True:
        break
    
    # if epoch > 5:
    #     learning_rate = learning_rate / 1.6
    #     teacher_forcing_ratio = teacher_forcing_ratio / 1.2
    # scheduler_enc.step()
    # scheduler_dec.step()
    print('######## epoch:{} ########'.format(epoch))
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    print_loss_total = 0
    for i, (input_sentences, target_sentences) in enumerate(train_loader):
        if EarlyTermination == True:
            break
        encoder.train()
        decoder.train()
        encoder.zero_grad()
        decoder.zero_grad()

        loss = 0

        encoder_hidden = encoder.initHidden(input_sentences.size()[0])
        encoder_outputs, encoder_hidden = encoder(input_sentences.to(device), encoder_hidden)

        encoder_outputs = encoder_outputs.to(device)
        decoder_input = SOS_token * torch.ones(input_sentences.size()[0], dtype = torch.long, device=device)
        decoder_input = decoder_input.unsqueeze(1)
        decoder_hidden = encoder_hidden.to(device)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for j in range(target_sentences.size()[1]):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(device), decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_sentences[:, j].to(device))
                decoder_input = target_sentences[:, j]
                decoder_input = decoder_input.unsqueeze(1)# Teacher forcing
        else:
            for j in range(target_sentences.size()[1]):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(device), decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_sentences[:, j].to(device))
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()
                #decoder_input = decoder_input# detach from history as input

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        print_loss_total += loss.item()/target_sentences.size()[1]

        if i > 0 and i % print_every == 0:
            step_record.append( i + epoch * total_step)
            print('%s (%d %d%%) %.4f' % (timeSince(start, (i+epoch*total_step) /(total_step*EPOCH)), \
                                         i+epoch*total_step, ((i+epoch*total_step) /(total_step*EPOCH))*100, \
                                         print_loss_total/print_every))
            train_loss_record.append([i+epoch*total_step, print_loss_total/print_every])
            print_loss_total = 0
        if i > 0 and i % val_every == 0:
            val_loss = dataset_loss(encoder, decoder, val_loader)
            predict_sentences = evaluate(val_loader, encoder, decoder, output_lang)
            val_Bleu = raw_corpus_bleu(predict_sentences, [dev_en]).score
            print('step: {}, valNLLLoss: {}, valBleuScore: {}'.format(i+epoch*total_step, val_loss, val_Bleu))
            val_loss_record.append([i+epoch*total_step, val_loss, val_Bleu])
            if epoch > 0 and val_Bleu>max_Bleu:
                torch.save(encoder.state_dict(), res_dataDir_cn + \
                           "//zh_tmp_encoder_1210_simple_t2_"+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+".pth")
                torch.save(decoder.state_dict(), res_dataDir_cn + \
                           "//zh_tmp_attn_decoder_1210_simple_t2_"+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+".pth")
            max_Bleu = max(max_Bleu, val_Bleu)
            print('Cache memo after Eva:{}'.format(torch.cuda.memory_allocated()))
            # Early Termination
            if epoch > 0 and val_loss_record[-1][2] < max_Bleu-3 and \
                                        val_loss_record[-2][2] < max_Bleu-2 and \
                                        val_loss_record[-1][2] < val_loss_record[-2][2]:
                EarlyTermination = True

pkl_dumper(train_loss_record, res_dataDir_cn + '//zh_trainLoss_1210_simple_t2_'+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+'.p')
pkl_dumper(val_loss_record, res_dataDir_cn + '//zh_valLoss_1210_simple_t2_'+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+'.p')
torch.save(encoder.state_dict(), res_dataDir_cn + "//zh_encoder_batch_1210_simple_t2_"+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+".pth")
torch.save(decoder.state_dict(), res_dataDir_cn + "//zh_attn_decoder_batch_1210_simple_t2_"+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+".pth")
