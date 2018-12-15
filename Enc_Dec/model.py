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

from sacrebleu import corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, pre_zh_emb_matrix):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=0)
        ## Load pretrained emb
        self.embedding.weight.data.copy_(torch.from_numpy(pre_zh_emb_matrix))
        self.gru = nn.GRU(emb_size, hidden_size, self.num_layers, batch_first=True, bidirectional = True)

    def forward(self, wordIn, hidden):
        # embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding(wordIn)
        output = embedded
        output, hidden = self.gru(output, hidden)
        # return output.cuda(), hidden.cuda()
        return output.to(device), hidden.to(device)

    def initHidden(self, batch_size):
        return torch.zeros(2*self.num_layers, batch_size, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, emb_size,  hidden_size, vocab_size, pre_en_emb_matrix):
        super(DecoderRNN, self).__init__()
        self.num_layers = 2
        self.bidirectional = False
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx = 0)
        self.embedding.weight.data.copy_(torch.from_numpy(pre_en_emb_matrix))
        self.gru = nn.GRU(emb_size, hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.linear = nn.Linear((self.bidirectional+1)*hidden_size, vocab_size) ## The output of GRU has the same length as sentence length.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, s1, hidden):
        embedded = self.embedding(s1)
        embedded = F.relu(embedded)
        #embedded = embedded.view(1,1,-1)
        rnn_out, hidden = self.gru(embedded, hidden)
        rnn_out = rnn_out.squeeze(1)
        rnn_out = self.linear(rnn_out)
        rnn_out = self.softmax(rnn_out.squeeze(1))
        return rnn_out, hidden

class Attn(nn.Module):
    def __init__(self, hidden_size, method='concat'):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        # self.emb_size = emb_size
        #self.attn = nn.Linear(self.hidden_size*2, hidden_size)
        self.attn = nn.Linear(self.hidden_size*3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)
        # print('mLen:{}, batSize:{}'.format(max_len,this_batch_size))
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        #encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return F.softmax(attn_energies, 2).to(device) # normalize with softmax [B*1*T]

    def score(self, hidden, encoder_outputs):
        #print('hl size:{}, eo size:{}'.format(hidden.size(), encoder_outputs.size()))
        #energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs.transpose(0,1)], 2))) # [B*T*2H]->[B*T*H]
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(energy.size()[0],1).unsqueeze(0).transpose(0,1) #[1*B*H]
        # print('vSize:{}, energySize:{}'.format(v.size(), energy.size()))
        #print('v size:{}, eng size:{}'.format(v.size(), energy.size()))
        energy = torch.bmm(v,energy) # [B*1*T]
        #return energy.squeeze(1) #[B*T]
        return energy.to(device)

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pre_en_emb_matrix, n_layers=2, dropout_p=0.1, max_len = 100):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_len = max_len
        self.bidirectional = False

        # Define layers
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=0)
        ## Load pretrained emb
        self.embedding.weight.data.copy_(torch.from_numpy(pre_en_emb_matrix))

        self.dropout = nn.Dropout(dropout_p)
        # self.attn = nn.Linear(hidden_size, self.max_len)
        self.attn = Attn(hidden_size)
        self.gru = nn.GRU(emb_size + 2*hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p, bidirectional = self.bidirectional) #Multiply 2 here since the encoder is bidirestional
        self.out = nn.Linear((self.bidirectional +1)*hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs

        # Get the embedding of the current input word (last output word)
        #print('wi size:{}'.format(word_input.size()))
        word_embedded = self.embedding(word_input).view(1, word_input.size()[0], -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)
        attn_weights = self.attn(last_hidden[-1], encoder_outputs )
        # print('wt size:{}, eo size:{}'.format(attn_weights.size(), encoder_outputs.size()))
        # attn_applied = attn_weights.bmm(encoder_outputs.unsqueeze(0))
        attn_applied = attn_weights.bmm(encoder_outputs)# (B,1,V)
        attn_applied = attn_applied.transpose(0, 1)
        # print('we size:{}, lh size:{}, aa size:{}'.format(word_embedded.size(), last_hidden.size(), attn_applied.size()))
        rnn_input = torch.cat((word_embedded, attn_applied), 2)
        rnn_input = rnn_input.transpose(0, 1)
        #print('rnn_input shape ', rnn_input.shape)
        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(1) # B x N
        #print('output of rnn shape ', output.size())
        output = F.log_softmax(self.out(output))

        # Return final output, hidden state, and attention weights (for visualization)
        return output.to(device), hidden.to(device), attn_weights.to(device)
