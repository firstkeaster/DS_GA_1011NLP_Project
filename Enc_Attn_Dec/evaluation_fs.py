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

def evaluate(loader, encoder, decoder, output_lang, MAX_LENGTH = 100):
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
            predicted_sentence = ''

            for di in range(MAX_LENGTH):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(device), decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token or topi.item() == PAD_token:
                    # predicted_sentence += '<eos>'
                    predicted_sentence += ''
                    break
                else:
                    #predicted_sentence.append(output_lang.index2word[topi.item()])
                    predicted_sentence += output_lang.index2word[topi.item()]
                    predicted_sentence += " "

                decoder_input = topi.detach()
                #decoder_input = decoder_input.unsqueeze(0)
            if di==MAX_LENGTH -1:
                # predicted_sentence += '<eos>'
                predicted_sentence += ''

            sentences_list.append(predicted_sentence)
            #decoder_attentions_list.append(decoder_attentions[:di + 1])

        #return decoded_words_list, decoder_attentions_list
    return sentences_list
