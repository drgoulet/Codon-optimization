# Train a recurrent neural network for codon optimization in CHO cells
# Based on Chinese hamster DNA and AA sequences
# Dennis R. Goulet
# First upload to Github: 03 July 2020

import os
import json
import io
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Encrypt DNA and AA sequences into individuals words (codons and residues) by adding spaces
def encrypt(string,length):
    return ' '.join(string[i:i+length] for i in range(0,len(string),length))

# Tokenize DNA and AA sequences
def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer

# Pad sequences if they are shorter than the max sequence length
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

# Combine tokenization and padding into one preprocessing function
def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

# Add desired layers to RNN model
def model_embed(input_shape, output_sequence_length, aa_vocab_size, dna_vocab_size):
    learning_rate = 0.005
    model = Sequential()
    model.add(Embedding(aa_vocab_size, 128, input_length=input_shape[1], input_shape=input_shape[1:],mask_zero=True))
    model.add(Bidirectional(GRU(16, return_sequences=True)))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(dna_vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(learning_rate), metrics=['accuracy'])

    return model

# cd to appropriate directory
os.chdir('/mnt/c/RNN')

# Read in dictionary of matched DNA, AA sequences
with open('chinese_hamster_dictionary.json') as f:
    dna_aa_dict = json.load(f)

# Make separate lists for the 30,000 DNA, AA training sequences (previously randomized)
dna_list_pre = list(dna_aa_dict.keys())
aa_list_pre = list(dna_aa_dict.values())
dna_list = dna_list_pre[:30000]
aa_list = aa_list_pre[:30000]

# Encrypt DNA, AA sequences into separate 'words' by adding spaces every 3 or 1 characters
aa_spaces = []
for aa_seq in aa_list:
    aa_current = encrypt(aa_seq,1)
    aa_spaces.append(aa_current)
dna_spaces = []
for dna_seq in dna_list:
    dna_current = encrypt(dna_seq,3)
    dna_spaces.append(dna_current)

# Preprocess DNA and AA sequences (tokenize and pad)
preproc_aa, preproc_dna, aa_tokenizer, dna_tokenizer = preprocess(aa_spaces, dna_spaces)

# Export DNA and AA tokenizers, which will be used when predicting codon optimized sequence
aa_tokenizer_json = aa_tokenizer.to_json()
with io.open('aa_tokenizer.json','w', encoding='utf-8') as f:
    f.write(json.dumps(aa_tokenizer_json, ensure_ascii=False))

dna_tokenizer_json = dna_tokenizer.to_json()
with io.open('dna_tokenizer.json','w', encoding='utf-8') as f:
    f.write(json.dumps(dna_tokenizer_json, ensure_ascii=False))

# Ensure correct dimensionality
tmp_x = pad(preproc_aa, preproc_dna.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_dna.shape[-2]))

# Make RNN model as defined previously, and using dimensions of AA, DNA vectors
my_model = model_embed(tmp_x.shape, preproc_dna.shape[1], len(aa_tokenizer.word_index)+1, len(dna_tokenizer.word_index)+1)

# Show the model parameters
my_model.summary()

# Allow for early stopping if the loss plateus, and export the newest model each epoch
callbacks = [EarlyStopping(monitor='val_loss', patience=5), ModelCheckpoint('rnn_model_newest.h5')]

# Fit the RNN model to DNA, AA sequences, and store the history of loss/accuracy
hist = my_model.fit(tmp_x, preproc_dna, batch_size=16, epochs=10, validation_split=0.2, callbacks=callbacks)

# Save the model for use in prediction, evaluation
my_model.save('rnn_model.h5')

# Also export the history of loss/accuracy
with open('rnn_history','wb') as f:
    pickle.dump(hist.history, f)
