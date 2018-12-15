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
from math import exp

from sacrebleu import raw_corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

def dataset_loss(encoder, decoder, dataloader, criterion = nn.NLLLoss().cuda()):
    encoder.eval()
    decoder.eval()
    loss = 0
    total = 0
    with torch.no_grad():
        for i, (input_sentences, target_sentences) in enumerate(dataloader):

            encoder_hidden = encoder.initHidden(input_sentences.size()[0]).cuda()
            encoder_outputs, encoder_hidden = encoder(input_sentences.to(device), encoder_hidden.to(device))
            encoder_outputs = encoder_outputs.to(device)

            decoder_input = SOS_token * torch.ones(input_sentences.size()[0], dtype = torch.long, device=device)
            decoder_input = decoder_input.unsqueeze(1)
            decoder_hidden = encoder_hidden.to(device)

            for j in range(target_sentences.size()[1]):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(device), decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, target_sentences[:, j].to(device))
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()

            total += target_sentences.size()[1]

    return loss/total

def evaluate_BeamSearch(loader, encoder, decoder, output_lang, MAX_LENGTH = 100, K_BS=4):
    def Beam_Search_Dec_Iterator(MAX_LENGTH, decoder_input, decoder_hidden, encoder_outputs, K_BS):
    ## K_BS: select K max possible values in candidates
        last_candidates = [[1, [decoder_input], decoder_hidden]]
        out_candidates = []
        min_proba = 1
        for di in range(MAX_LENGTH):
            cur_candidates = []
            if len(last_candidates) <= 0:
                break
            for proba, cur_centense, decoder_hidden in last_candidates:
                decoder_input = cur_centense[-1].detach()
                # print(decoder_input)
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(device), decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(K_BS)
                # print([exp(x.item()) for x in topv[0]], topi)
                for j, v in enumerate(topi[0]):
                    # print()
                    this_cen = cur_centense+[v.unsqueeze(0).unsqueeze(0)]
                    # this_proba = proba*(exp(topv[0][j].item()))
                    this_proba = proba*exp(topv[0][j].item())
                    this_hidden = decoder_hidden
                    if di == MAX_LENGTH-1:
                        this_cen += [torch.tensor([[EOS_token]])]
                        out_candidates.append([this_proba, this_cen])
                    if v.item() == EOS_token or v.item() == PAD_token:
                        this_cen += [torch.tensor([[EOS_token]])]
                        out_candidates.append([this_proba, this_cen])
                        # min_proba = min(min_proba, this_proba)
                        # K_BS -= 1
                    else:
                        cur_candidates.append([this_proba, this_cen, this_hidden])
            # print(cur_candidates[0])
            cur_candidates = sorted(cur_candidates, key=lambda x:-x[0])[:K_BS]
            last_candidates = cur_candidates[:K_BS]

        out_candidates = sorted(out_candidates, key=lambda x:-x[0])[:K_BS]
        out_sentences = []
        for _, can in out_candidates:
            # print(_)
            out_sentences.append(' '.join([output_lang.index2word[v.item()] for v in can]).strip('SOS EOS'))

        return(out_sentences[0])

    encoder.eval()
    decoder.eval()
    sentences_list = []

    with torch.no_grad():
        for i, (input_sentence, output_sentence) in enumerate(loader):
            encoder_hidden = encoder.initHidden(input_sentence.size()[0]).cuda()
            encoder_outputs, encoder_hidden = encoder(input_sentence.to(device), encoder_hidden)

            decoder_input = torch.tensor(np.array([[SOS_token]]), device=device)
            decoder_hidden = encoder_hidden
            #predicted_sentence = []
            predicted_sentences = Beam_Search_Dec_Iterator(MAX_LENGTH, decoder_input, decoder_hidden, encoder_outputs, K_BS)

            sentences_list.append(predicted_sentences)

    return sentences_list