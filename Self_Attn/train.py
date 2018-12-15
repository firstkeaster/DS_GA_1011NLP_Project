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
import math, copy, time

from sacrebleu import corpus_bleu

from utils import *
from model import *
from evaluation import *

BATCH_SIZE = 64
EMB_SIZE = 300
HIDDEN_SIZE = 720
EPOCH = 10
IF_SORT = True
IF_SHUFFLE = False
#MAX_LEN_filter = 50
MAX_LEN_filter = 37
MAX_LEN = MAX_LEN_filter
d_model = HIDDEN_SIZE
d_ff = 2048
h = 8
N = 6

lang_name = 'vi'

print('Input language ', lang_name)
print('Self attention encoder Model with BATCH_SIZE ', BATCH_SIZE, ' HIDDEN_SIZE ', HIDDEN_SIZE, ' EPOCH ', EPOCH, ' h', h)
print('If sort ', IF_SORT)
print('MAX filter length ', MAX_LEN_filter)

teacher_forcing_ratio = 1
learning_rate = 0.0001
print_every = 500
val_every = 1000

nowpath = os.path.abspath('.')
zh_en_dataDir = nowpath + '/project/iwslt-'+lang_name+'-en'
#vi_en_dataDir = nowpath + '//dataset//iwsltvien'
#emb_dataDir = nowpath + '//dataset//embedding'
emb_dataDir = nowpath + '/project/'
res_dataDir = nowpath + '/project/result'

"""=============================================================================="""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

train_en = []
zh_en_reader(train_en, zh_en_dataDir + '//train.tok.en')
train_zh = []
zh_en_reader(train_zh, zh_en_dataDir + '//train.tok.'+lang_name)
dev_zh = []
zh_en_reader(dev_zh, zh_en_dataDir + '//dev.tok.'+lang_name)
dev_en = []
zh_en_reader(dev_en, zh_en_dataDir + '//dev.tok.en')
test_zh = []
zh_en_reader(test_zh, zh_en_dataDir + '//test.tok.'+lang_name)
test_en = []
zh_en_reader(test_en, zh_en_dataDir + '//test.tok.en')

data_preprocessor(dev_zh)
data_preprocessor(dev_en)
data_preprocessor(train_en)
data_preprocessor(train_zh)
data_preprocessor(test_zh)
data_preprocessor(test_en)

input_lang, output_lang, pairs = prepareData_zh_en_ensemble(lang_name, 'en', train_zh, train_en)
input_train, output_train = ConvertSentence2numpy(input_lang, output_lang, pairs, UNK_token, EOS_token, MAX_LEN_filter)
input_test, output_test = ConvertSentence2numpy(input_lang, output_lang, list(zip(test_zh, test_en)), UNK_token, EOS_token, MAX_LEN_filter)
input_dev, output_dev = ConvertSentence2numpy(input_lang, output_lang, list(zip(dev_zh, dev_en)), UNK_token, EOS_token, MAX_LEN_filter)
test_en_filter = filter_target(input_lang, list(zip(test_zh, test_en)), MAX_LEN_filter, UNK_token)
dev_en_filter = filter_target(input_lang, list(zip(dev_zh, dev_en)), MAX_LEN_filter, UNK_token)

print('Filter length is ', MAX_LEN_filter,' ', len(input_train)/len(train_zh), ' remains after filter.')
pre_zh_emb_matrix = Load_pretrained_emb(10, input_lang, emb_dataDir)
pre_en_emb_matrix = Load_pretrained_emb(30000, output_lang, emb_dataDir)

train_dataset = TransDataset(input_train, output_train, IF_SORT)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=IF_SHUFFLE)

val_dataset = TransDataset(input_dev, output_dev, True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=False)
val_loader1 = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=1,
                                           collate_fn=vocab_collate_func,
                                           shuffle=False)

#encoder = EncoderRNN(input_lang.n_words, EMB_SIZE, HIDDEN_SIZE, pre_zh_emb_matrix).to(device)
Self_Attn = MultiHeadedAttention( h, d_model).to(device)
feed_forward = PositionwiseFeedForward(d_model, d_ff).to(device)
layer = EncoderLayer(d_model, Self_Attn, feed_forward, 0.1).to(device)
encoder = Encoder(input_lang.n_words, d_model, d_model, layer, N).to(device)
decoder = BahdanauAttnDecoderRNN(EMB_SIZE, HIDDEN_SIZE, output_lang.n_words, pre_en_emb_matrix).to(device)
#decoder = AttnDecoderRNN(EMB_SIZE, HIDDEN_SIZE, output_lang.n_words, pre_en_emb_matrix, MAX_LEN).to(device)
#decoder = BahdanauAttnDecoderRNN(EMB_SIZE, HIDDEN_SIZE, output_lang.n_words, pre_en_emb_matrix).to(device)
#decoder = DecoderRNN(EMB_SIZE, HIDDEN_SIZE, output_lang.n_words, pre_en_emb_matrix).to(device)
print('num_parameters_encoder:', sum(p.numel() for p in encoder.parameters()))
print('num_parameters_decoder:', sum(p.numel() for p in decoder.parameters()))

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

for epoch in range(EPOCH):
    if EarlyTermination == True:
        break
    learning_rate = learning_rate / 1.05
    teacher_forcing_ratio = teacher_forcing_ratio / 1.1
    print('######## epoch:{} ########'.format(epoch))
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    print_loss_total = 0
    for i, (input_sentences, target_sentences) in enumerate(train_loader):
        encoder.train()
        decoder.train()
        encoder.zero_grad()
        decoder.zero_grad()

        loss = 0

        #encoder_hidden = encoder.initHidden(input_sentences.size()[0])
        #encoder_outputs, encoder_hidden = encoder(input_sentences.to(device), encoder_hidden)
        # encoder_outputs = torch.zeros(input_sentences.size()[0], MAX_LEN, 2*HIDDEN_SIZE)
        # a, encoder_hidden = encoder(input_sentences.to(device), None)
        # encoder_outputs[:, :input_sentences.size()[1], :] = a
        #encoder_outputs, encoder_hidden = encoder(input_sentences.to(device), None)
        encoder_outputs = encoder(input_sentences.to(device), None)
        encoder_hidden = encoder_outputs[:,-2:,:].transpose(0,1).contiguous()

        encoder_outputs = encoder_outputs.to(device)
        decoder_input = SOS_token * torch.ones(input_sentences.size()[0], dtype = torch.long, device=device)
        decoder_input = decoder_input.unsqueeze(1)
        decoder_hidden = encoder_hidden.to(device)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for j in range(target_sentences.size()[1]):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(device), decoder_hidden, encoder_outputs)
                #decoder_output, decoder_hidden = decoder(decoder_input.to(device), decoder_hidden)
                loss += criterion(decoder_output, target_sentences[:, j].to(device))
                decoder_input = target_sentences[:, j]
                decoder_input = decoder_input.unsqueeze(1)# Teacher forcing
        else:
            for j in range(target_sentences.size()[1]):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(device), decoder_hidden, encoder_outputs)
                #decoder_output, decoder_hidden = decoder(decoder_input.to(device), decoder_hidden)
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
            predict_sentences = evaluate(val_loader1, encoder, decoder, output_lang)
            val_Bleu = corpus_bleu(predict_sentences, [dev_en_filter]).score
            #max_Bleu = max(val_Bleu, max_Bleu)
            if val_Bleu > max_Bleu:
                max_Bleu = val_Bleu
                torch.save(encoder.state_dict(), res_dataDir + "//" + lang_name+ "_encoder_selfattn_"+str(IF_SORT)+"_"+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+'_'+str(h)+".pth")
                torch.save(decoder.state_dict(), res_dataDir + "//" +lang_name + "_attn_selfattn_"+str(IF_SORT)+"_"+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+'_'+str(h)+".pth")
#             valNLL = all_dataset_loss_batch(encoder, decoder, valPairs)
#             valBleu = all_dataset_BLEUScoreLoss_batch(encoder, decoder, valPairs, bleuFunc = get_moses_multi_bleu)
            print('step: {}, valNLLLoss: {}, valBleuScore: {}'.format(i+epoch*total_step, val_loss, val_Bleu))
            val_loss_record.append([i+epoch*total_step, val_loss, val_Bleu])
            # Early Termination
            # if len(val_loss_record) > 2 and val_loss_record[-1][2] < max_Bleu-0.5:
            #     EarlyTermination = True

pkl_dumper(train_loss_record, res_dataDir + '//'+lang_name+'_trainLoss_selfattn_'+str(IF_SORT)+'_'+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+'_'+str(h)+'.p')
pkl_dumper(val_loss_record, res_dataDir + '//'+lang_name+'_valLoss_selfattn_'+str(IF_SORT)+'_'+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+'_'+str(h)+'.p')
# torch.save(encoder.state_dict(), res_dataDir + "//" + lang_name+ "_encoder_selfattn_"+str(IF_SORT)+"_"+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+'_'+str(h)+".pth")
# torch.save(decoder.state_dict(), res_dataDir + "//" +lang_name + "_attn_selfattn_"+str(IF_SORT)+"_"+str(EMB_SIZE)+'_'+str(HIDDEN_SIZE)+'_'+str(EPOCH)+'_'+str(h)+".pth")
