# DS_GA_1011NLP_Project

This project introduces several approaches for implementation of neural machine translator (NMT). We tried different structures of the NMT, including RNN based encoder-decoder, RNN encoder-Global attention-RNN decoder, and self-attention based encoder models. 
To boost the performance, we tried several different structures of the attention method, implemented loader with minibatch and built beam-search algorithm based predictor. We trained our model on vi-en and zh-en corpus. We got BLEU score of 18.52 with our vi-en model and 12.50 with our zh-en model (evaluation).

For the Enc_Attn_Dec part, our attention model in "model.py" file, we referred this repository: https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq. We changed its structure, and didn't use the model in our final version. We only used models in model_new and model_new_simple, which is our original work, for HP tuning and evaluation.

For the Self_Attn part, the self attention encoder model in "model.py" file, we referred the link http://nlp.seas.harvard.edu/2018/04/03/attention.html
