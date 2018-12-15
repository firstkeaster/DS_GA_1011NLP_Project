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

from sacrebleu import raw_corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, pre_zh_emb_matrix=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=0)
        ## Load pretrained emb
        if pre_zh_emb_matrix is not False:
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

class NewAttn(nn.Module):
    ## A dot product attention, applied after the GRU layer in decoder.
    def __init__(self, hidden_size, method='dot'):
        super(NewAttn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        # self.attn = nn.Linear(self.hidden_size, self.hidden_size)
    def forward(self, this_output, encoder_outputs):
        # enc_trans = self.attn(encoder_outputs)
        weights = F.softmax(torch.bmm(this_output, encoder_outputs.transpose(1,2)), 2) ##B*1*H, B*H*T
        return weights.to(device) #B*1*T

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pre_en_emb_matrix=False, n_layers=2, dropout_p=0.2, max_len = 100):
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
        if pre_en_emb_matrix is not False:
            self.embedding.weight.data.copy_(torch.from_numpy(pre_en_emb_matrix))
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p, bidirectional = self.bidirectional) 
        self.attn = NewAttn(hidden_size)
        self.h_out = nn.Linear((self.bidirectional +1)*2*hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
        
        word_embedded = self.embedding(word_input) #.view(1, word_input.size()[0], -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)
        this_output, hidden = self.gru(word_embedded, last_hidden)
        encoder_outputs = encoder_outputs.view(encoder_outputs.size()[0], encoder_outputs.size()[1], 2, -1)[:, :, 0]
        attn_weights = self.attn(this_output, encoder_outputs) #(B,1,T)
        attn_applied = attn_weights.bmm(encoder_outputs)# (B, 1, H)
        output = torch.cat((attn_applied.squeeze(1), this_output.squeeze(1)), 1)
        output = F.tanh(self.h_out(output))
        output = F.log_softmax(self.out(output))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output.to(device), hidden.to(device), attn_weights.to(device)
