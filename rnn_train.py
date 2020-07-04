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

os.chdir('/mnt/c/python_work/')

with open('cho_dict.json') as f:
    my_dict = json.load(f)

dna_list_pre = list(my_dict.keys())
aa_list_pre = list(my_dict.values())

dna_list = dna_list_pre[:20]
aa_list = aa_list_pre[:20]

# aa_test = aa_list_pre[0]

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

print(aa_spaces)
print(dna_spaces)

# aa_space_length = max([len(sequence) for sequence in aa_spaces])
# print(aa_space_length)

# dna_space_length = max([len(sequence) for sequence in dna_spaces])
# print(dna_space_length)

# for sample_i in range(5):
#     print('English sample {}:  {}'.format(sample_i + 1, dna_spaces[sample_i]))
#     print('French sample {}:  {}\n'.format(sample_i + 1, aa_spaces[sample_i]))

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

# print(aa_spaces,dna_spaces)

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def model_embed(input_shape, output_sequence_length, aa_vocab_size, dna_vocab_size):
    learning_rate = 0.005

    model = Sequential()
    model.add(Embedding(aa_vocab_size, 256, input_length=input_shape[1], input_shape=input_shape[1:]))
    model.add(Bidirectional(GRU(64, return_sequences=True)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(dna_vocab_size, activation='softmax')))

    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(learning_rate), metrics=['accuracy'])

    return model

tmp_x = pad(preproc_aa, preproc_dna.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_dna.shape[-2]))

# tmp_y = pad(preproc_dna[1:], preproc_dna.shape[1])

# x_pred = pad(preproc_aa[0], preproc_dna.shape[1])
# x_pred = x_pred.reshape((-1, preproc_dna.shape[-2]))

my_model = model_embed(tmp_x.shape, preproc_dna.shape[1], len(aa_tokenizer.word_index)+1, len(dna_tokenizer.word_index)+1)

my_model.summary()

callbacks = [EarlyStopping(monitor='val_loss', patience=2), ModelCheckpoint('keras_model_combined.h5')]

my_model.fit(tmp_x, preproc_dna, batch_size=2, epochs=10, validation_split=0.2, callbacks=callbacks)

print(logits_to_text(my_model.predict(tmp_x[:1])[0], dna_tokenizer))

my_model.save('keras_model_combined_final.h5')

