# Use a trained recurrent neural network to codon optimize an amino acid sequence
# Predicts the 'best' DNA sequence for a given AA squence
# Based on Chinese hamster DNA and AA sequences
# Inputs: Trained RNN model (.h5), DNA and AA tokenizers (.json), and AA sequence to optimize (.txt)
# Formatting of AA sequence: Single-letter abbreviations, no spaces, no stop codon (this is added), include signal peptide
# Output: predicted/optimized DNA sequence (.txt)
# Dennis R. Goulet
# First upload to Github: 03 July 2020

import os
import numpy as np
import json

from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Set working directory and place trained model and tokenizers here
os.chdir('/mnt/c/RNN/')

# Place text file with AA sequence to be optimized in working directory
# Set the name of the text file here:
input_seq = "sequence"

# Encrypt the amino acid sequence
def encrypt(string,length):
    return ' '.join(string[i:i+length] for i in range(0,len(string),length))

# Pad the amino acid sequence to the correct length (matching model)
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

# Combine tokenization and padding
def preprocess(x):
    preprocess_x = aa_tokenizer.texts_to_sequences(x)
    preprocess_x = pad(preprocess_x)
    return preprocess_x

# Transform tokens back to DNA sequence
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

# Import sequence to optimize/predict from text file
aa_seq_file = input_seq + ".txt"
with open(aa_seq_file) as f:
    aa_item = f.read()

# Import trained model as .h5    
model = load_model('rnn_model.h5')

# Import tokenizers as json
with open('aa_tokenizer.json') as f:
    aa_json = json.load(f)
aa_tokenizer = tokenizer_from_json(aa_json)

with open('dna_tokenizer.json') as f:
    dna_json = json.load(f)
dna_tokenizer = tokenizer_from_json(dna_json)

# Add stop codon identifier and calculate length of input sequence
aa_item_with_stop = aa_item + 'Z'
aa_list = [aa_item_with_stop]
seq_len = len(aa_item_with_stop)

# Encrypt the input amino acid sequence by adding spaces
aa_spaces = []
for aa_seq in aa_list:
    aa_current = encrypt(aa_seq,1)
    aa_spaces.append(aa_current)

# Tokenize the amino acid sequence
# If using a different model, need to change dimensions accordingly
preproc_aa = preprocess(aa_spaces)
tmp_x = pad(preproc_aa, 8801)
tmp_x = tmp_x.reshape((-1, 8801))

# Use the imported model to predict the best DNA sequence for the input AA sequence
# Format the result to be a string of nucleotides (DNA sequence)
seq_opt = logits_to_text(model.predict(tmp_x[:1])[0], dna_tokenizer)
seq_opt_removepad = seq_opt[:(seq_len*4)]
seq_opt_removespace = seq_opt_removepad.replace(" ","")
seq_opt_final = seq_opt_removespace.upper()

# Print the predicted/optimized DNA sequence
print("Optimized DNA sequence:")
print(seq_opt_final)

# Export optimized/predicted DNA sequence as text file
with open(input_seq+"_opt.txt", "w") as f:
    print(seq_opt_final, file=f)
