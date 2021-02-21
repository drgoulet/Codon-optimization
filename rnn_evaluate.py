# Evaluate the accuracy of a trained recurrent neural network
# Imports: trained model (.h5), AA/DNA tokenizers (.json), test set as dictionary (.json)
# Export: Loss, accuracy
# Dennis R. Goulet
# First upload to Github: 21 February 2021

import os
import json
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Encrypt DNA and AA sequences into individuals words (codons and residues) by adding spaces
def encrypt(string,length):
    return ' '.join(string[i:i+length] for i in range(0,len(string),length))

# Pad sequences if they are shorter than the max sequence length
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

# Combine tokenization and padding
# Used 8801 based on dimensions of model
def preprocess(x,y):
    preprocess_x = aa_tokenizer.texts_to_sequences(x)
    preprocess_x = pad(preprocess_x, 8801)
    preprocess_y = dna_tokenizer.texts_to_sequences(y)
    preprocess_y = pad(preprocess_y, 8801)
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y

# Set the working directory
os.chdir('/mnt/c/RNN')

# Import trained model as .h5    
model = load_model('rnn_model.h5')

# Read in dictionary of matched DNA, AA sequences (contains test set)
with open('chinese_hamster_dictionary.json') as f:
    dna_aa_dict = json.load(f)

# Make separate lists for the 8,000 DNA, AA test sequences
dna_list_pre = list(dna_aa_dict.keys())
aa_list_pre = list(dna_aa_dict.values())
dna_list = dna_list_pre[30001:38001]
aa_list = aa_list_pre[30001:38001]

# Encrypt DNA, AA sequences into separate 'words' by adding spaces every 3 or 1 characters
aa_spaces = []
for aa_seq in aa_list:
    aa_current = encrypt(aa_seq,1)
    aa_spaces.append(aa_current)
dna_spaces = []
for dna_seq in dna_list:
    dna_current = encrypt(dna_seq,3)
    dna_spaces.append(dna_current)

# Import tokenizers as json (must be same tokenizers from training)
with open('aa_tokenizer.json') as f:
    aa_json = json.load(f)
aa_tokenizer = tokenizer_from_json(aa_json)

with open('dna_tokenizer.json') as f:
    dna_json = json.load(f)
dna_tokenizer = tokenizer_from_json(dna_json)

# Preprocess DNA and AA sequences (tokenize and pad)
preproc_aa, preproc_dna = preprocess(aa_spaces, dna_spaces)

# Ensure correct dimensionality
tmp_x = pad(preproc_aa, preproc_dna.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_dna.shape[-2]))

# Evaluate the test sequences on the trained model
results = model.evaluate(preproc_aa,preproc_dna, batch_size=16)

# Export the evaluation loss and accuracy
with open('rnn_evaluation','wb') as f:
    pickle.dump(results, f)