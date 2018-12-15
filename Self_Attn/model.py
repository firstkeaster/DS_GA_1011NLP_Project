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
import math, copy, time

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

"""=================================================================================================================="""
#
# class AttnDecoderRNN(nn.Module):
#     def __init__(self, emb_size, hidden_size, output_size, pre_en_emb_matrix, max_length, dropout_p=0.1):
#         super(AttnDecoderRNN, self).__init__()
#         self.emb_size = emb_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#         self.n_layer = 2
#         self.bidirectional = False
#
#         self.embedding = nn.Embedding(self.output_size, self.emb_size, padding_idx = 0)
#         self.embedding.weight.data.copy_(torch.from_numpy(pre_en_emb_matrix))
#         self.attn = nn.Linear(self.emb_size + self.hidden_size, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2 + self.emb_size, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layer, batch_first=True, bidirectional = self.bidirectional )
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input)
#         embedded = self.dropout(embedded)
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded, hidden.sum(0, keepdim=True).transpose(1,0)), 2)), dim=1)
#         attn_applied = torch.bmm(attn_weights,
#                                  encoder_outputs).squeeze(1)
#
#         output = torch.cat((embedded.squeeze(1), attn_applied), 1)
#         output = self.attn_combine(output)
#
#         output = F.relu(output)
#         output, hidden = self.gru(output.unsqueeze(1), hidden)
#         output = F.log_softmax(self.out(output.squeeze(1)), dim=1)
#         return output, hidden, attn_weights
#
# """=================================================================================================================="""
#
# class Attn(nn.Module):
#     def __init__(self, hidden_size, method='concat'):
#         super(Attn, self).__init__()
#         self.method = method
#         self.hidden_size = hidden_size
#         # self.emb_size = emb_size
#         #self.attn = nn.Linear(self.hidden_size*2, hidden_size)
#         self.attn = nn.Linear(self.hidden_size*3, hidden_size)
#         self.v = nn.Parameter(torch.rand(hidden_size))
#         stdv = 1. / math.sqrt(self.v.size(0))
#         self.v.data.normal_(mean=0, std=stdv)
#
#     def forward(self, hidden, encoder_outputs):
#         '''
#         :param hidden:
#             previous hidden state of the decoder, in shape (layers*directions,B,H)
#         :param encoder_outputs:
#             encoder outputs from Encoder, in shape (T,B,H)
#         :return
#             attention energies in shape (B,T)
#         '''
#         max_len = encoder_outputs.size(1)
#         this_batch_size = encoder_outputs.size(0)
#         # print('mLen:{}, batSize:{}'.format(max_len,this_batch_size))
#         H = hidden.repeat(max_len,1,1).transpose(0,1)
#         #encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
#         attn_energies = self.score(H,encoder_outputs) # compute attention score
#         return F.softmax(attn_energies, 2).to(device) # normalize with softmax [B*1*T]
#
#     def score(self, hidden, encoder_outputs):
#         #print('hl size:{}, eo size:{}'.format(hidden.size(), encoder_outputs.size()))
#         #energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs.transpose(0,1)], 2))) # [B*T*2H]->[B*T*H]
#         energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
#         energy = energy.transpose(2,1) # [B*H*T]
#         v = self.v.repeat(energy.size()[0],1).unsqueeze(0).transpose(0,1) #[1*B*H]
#         # print('vSize:{}, energySize:{}'.format(v.size(), energy.size()))
#         #print('v size:{}, eng size:{}'.format(v.size(), energy.size()))
#         energy = torch.bmm(v,energy) # [B*1*T]
#         #return energy.squeeze(1) #[B*T]
#         return energy.to(device)
#
# class BahdanauAttnDecoderRNN(nn.Module):
#     def __init__(self, emb_size, hidden_size, output_size, pre_en_emb_matrix, n_layers=2, dropout_p=0.1, max_len = 100):
#         super(BahdanauAttnDecoderRNN, self).__init__()
#
#         # Define parameters
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.dropout_p = dropout_p
#         self.max_len = max_len
#         self.bidirectional = False
#
#         # Define layers
#         self.embedding = nn.Embedding(output_size, emb_size, padding_idx=0)
#         ## Load pretrained emb
#         self.embedding.weight.data.copy_(torch.from_numpy(pre_en_emb_matrix))
#
#         self.dropout = nn.Dropout(dropout_p)
#         # self.attn = nn.Linear(hidden_size, self.max_len)
#         self.attn = Attn(hidden_size)
#         self.gru = nn.GRU(emb_size + 2*hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p, bidirectional = self.bidirectional) #Multiply 2 here since the encoder is bidirestional
#         self.out = nn.Linear((self.bidirectional +1)*hidden_size, output_size)
#
#     def forward(self, word_input, last_hidden, encoder_outputs):
#         # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
#
#         # Get the embedding of the current input word (last output word)
#         #print('wi size:{}'.format(word_input.size()))
#         word_embedded = self.embedding(word_input).view(1, word_input.size()[0], -1) # S=1 x B x N
#         word_embedded = self.dropout(word_embedded)
#         attn_weights = self.attn(last_hidden[-1], encoder_outputs )
#         # print('wt size:{}, eo size:{}'.format(attn_weights.size(), encoder_outputs.size()))
#         # attn_applied = attn_weights.bmm(encoder_outputs.unsqueeze(0))
#         attn_applied = attn_weights.bmm(encoder_outputs)# (B,1,V)
#         attn_applied = attn_applied.transpose(0, 1)
#         # print('we size:{}, lh size:{}, aa size:{}'.format(word_embedded.size(), last_hidden.size(), attn_applied.size()))
#         rnn_input = torch.cat((word_embedded, attn_applied), 2)
#         rnn_input = rnn_input.transpose(0, 1)
#         #print('rnn_input shape ', rnn_input.shape)
#         output, hidden = self.gru(rnn_input, last_hidden)
#
#         # Final output layer
#         output = output.squeeze(1) # B x N
#         #print('output of rnn shape ', output.size())
#         output = F.log_softmax(self.out(output))
#
#         # Return final output, hidden state, and attention weights (for visualization)
#         return output.to(device), hidden.to(device), attn_weights.to(device)

"""============= Self_attention encoder =============================================================================================="""

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, vob_size, emb_size, d_model, layer, N = 1):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.d_model = d_model
        self.norm = LayerNorm(layer.size)
        self.linear = nn.Linear(d_model, d_model)
        self.emb = nn.Embedding(vob_size, emb_size)

    def forward(self, x, mask):
        x = self.emb(x)
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        #hidden = self.linear(x).sum(dim= 1).view(2, x.size(0), int(0.5*self.d_model))
        return self.norm(x)#, hidden

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    weighted_attn = torch.matmul(p_attn, value)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
    #def forward(self, query, key, value):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        x = self.linears[-1](x)
        #return self.linears[-1](x)
        return x

# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, emb_size, hidden_size, pre_zh_emb_matrix):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = 1
#
#         self.embedding = nn.Embedding(input_size, emb_size, padding_idx=0)
#         ## Load pretrained emb
#         self.embedding.weight.data.copy_(torch.from_numpy(pre_zh_emb_matrix))
#         self.gru = nn.GRU(emb_size, hidden_size, self.num_layers, batch_first=True, bidirectional = True)
#
#     def forward(self, wordIn, hidden):
#         # embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.embedding(wordIn)
#         output = embedded
#         output, hidden = self.gru(output, hidden)
#         # return output.cuda(), hidden.cuda()
#         return output.to(device), hidden.to(device)
#
#     def initHidden(self, batch_size):
#         return torch.zeros(2*self.num_layers, batch_size, self.hidden_size, device=device)

class NewAttn(nn.Module):
    def __init__(self, hidden_size, method='dot'):
        super(NewAttn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
    def forward(self, this_output, encoder_outputs):
        enc_trans = self.attn(encoder_outputs)
        weights = F.softmax(torch.bmm(this_output, enc_trans.transpose(1,2)), 2) ##B*1*H, B*H*T
        return weights.to(device) #B*1*T

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pre_en_emb_matrix, n_layers=2, dropout_p=0.2, max_len = 100):
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
        encoder_outputs = encoder_outputs.view(encoder_outputs.size()[0], encoder_outputs.size()[1], 1, -1)[:, :, 0]
        attn_weights = self.attn(this_output, encoder_outputs) #(B,1,T)
        attn_applied = attn_weights.bmm(encoder_outputs)# (B, 1, H)
        output = torch.cat((attn_applied.squeeze(1), this_output.squeeze(1)), 1)
        output = F.tanh(self.h_out(output))
        output = F.log_softmax(self.out(output))

        # Return final output, hidden state, and attention weights (for visualization)
        return output.to(device), hidden.to(device), attn_weights.to(device)
