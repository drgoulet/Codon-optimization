# Predict the best DNA sequence for a given AA sequence
# Uses trained .h5 model

import collections
import os
import helper
import numpy as np
import json
import numpy

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

os.chdir('/mnt/c/python_work/')

model = load_model('keras_model_combined.h5')

print(model.summary())


with open('cho_dict.json') as f:
    my_dict = json.load(f)

dna_list_pre = list(my_dict.keys())
aa_list_pre = list(my_dict.values())

dna_list = dna_list_pre[:20000]
aa_list = aa_list_pre[:20000]

aa_length = max([len(sequence) for sequence in aa_list])
dna_length = max([len(sequence) for sequence in dna_list])

print(len(dna_list))
print(len(aa_list))
print(aa_length)
print(dna_length)

def encrypt(string,length):
    return ' '.join(string[i:i+length] for i in range(0,len(string),length))

aa_spaces = []
for aa_seq in aa_list:
    aa_current = encrypt(aa_seq,1)
    aa_spaces.append(aa_current)

dna_spaces = []
for dna_seq in dna_list:
    dna_current = encrypt(dna_seq,3)
    dna_spaces.append(dna_current)

aa_counter = collections.Counter([word for sentence in aa_spaces for word in sentence.split()])
dna_counter = collections.Counter([word for sentence in dna_spaces for word in sentence.split()])

print('{} aa words.'.format(len([word for sentence in aa_spaces for word in sentence.split()])))
print('{} unique aa words.'.format(len(aa_counter)))
print('10 most common aa words:')
print('"' + '" "'.join(list(zip(*aa_counter.most_common(10)))[0]) + '"')

print('{} dna words.'.format(len([word for sentence in dna_spaces for word in sentence.split()])))
print('{} unique dna words.'.format(len(dna_counter)))
print('10 most common dna words:')
print('"' + '" "'.join(list(zip(*dna_counter.most_common(10)))[0]) + '"')

def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer

def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_aa, preproc_dna, aa_tokenizer, dna_tokenizer = preprocess(aa_spaces, dna_spaces)

max_aa_len = preproc_aa.shape[1]
max_dna_len = preproc_dna.shape[1]
aa_size = len(aa_tokenizer.word_index)
dna_size = len(dna_tokenizer.word_index)

print("Max aa length:", max_aa_len)
print("Max dna length:", max_dna_len)
print("AA vocab size:", aa_size)
print("DNA vocab size:", dna_size)

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

tmp_x = pad(preproc_aa, preproc_dna.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_dna.shape[-2]))

print(len(tmp_x))
print(len(tmp_x[0]))
print(len(tmp_x[1]))

print(model.predict(tmp_x[:1])[0])

print(logits_to_text(model.predict(tmp_x[:1])[0], dna_tokenizer))

