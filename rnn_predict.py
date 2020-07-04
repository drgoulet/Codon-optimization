# Predict the best DNA sequence for a given AA sequence
# Uses trained .h5 model
# Input is list of AA sequences, output is optimized DNA sequences

import collections
import os
import helper
import numpy as np
import json
import numpy

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

os.chdir('/mnt/c/python_work/')

model = load_model('rnn_final_model.h5')

with open('aa_tokenizer.json') as f:
    aa_json = json.load(f)

aa_tokenizer = tokenizer_from_json(aa_json)

with open('dna_tokenizer.json') as f:
    dna_json = json.load(f)

dna_tokenizer = tokenizer_from_json(dna_json)

with open('IgG1.txt') as f:
    aa_item = f.read()

aa_list = [aa_item]

def encrypt(string,length):
    return ' '.join(string[i:i+length] for i in range(0,len(string),length))

aa_spaces = []
for aa_seq in aa_list:
    aa_current = encrypt(aa_seq,1)
    aa_spaces.append(aa_current)

def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def preprocess(x):
    preprocess_x = aa_tokenizer.texts_to_sequences(x)
    preprocess_x = pad(preprocess_x)

    return preprocess_x

preproc_aa = preprocess(aa_spaces)

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

tmp_x = pad(preproc_aa, 8801)
tmp_x = tmp_x.reshape((-1, 8801))

print(logits_to_text(model.predict(tmp_x[:1])[0], dna_tokenizer))
