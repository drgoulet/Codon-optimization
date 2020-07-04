import json
import numpy as np
import pandas as pd
from Bio import SeqIO 
import random
from collections import OrderedDict

# Purpose: Import fasta file containing list of DNA sequence, verify DNA and corresponding AA sequences, output verified sequences as json.
# Dennis R. Goulet
# 17 May 2020

# Initialize lists and define acceptable DNA bases and amino acids.
dna_seq = []
dna_seq_new = []
aa_seq_new = []
bases = ['A','C','G','T']
residues = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Z']

# Initialize list of tests to verify properties of the DNA and AA sequences.
test_total = [0,0,0,0,0,0,0]

# Import starting list of DNA sequences from fasta file containing human CDS sequences.
for sequence in SeqIO.parse('cds_hamster.fna','fasta'):
    dna_seq.append(sequence.seq.upper())

# Verify properties of DNA sequences and their resulting AA sequences. 
# If all tests pass (0), DNA and AA sequences are inserted into matching indices of two new lists. 
for sequence in dna_seq:

    # Re-initialize test scores. 
    test = [1,0,1,1,1,0,1]

    # The first test verifies that the length of the DNA sequence is divisible by 3, indicating it can be translated. 
    if len(sequence) % 3 == 0:
        test[0] = 0

        # The second test verifies that the DNA sequence only contains the 4 standard DNA bases.
        for base in sequence:
            if base not in bases:
                test[1] = 1

        # If both of the DNA sequence tests pass, the DNA sequence is translated to AA sequence.
        single_aa_seq_pre = str(sequence.translate())
        single_aa_seq = single_aa_seq_pre.replace('*','Z')

        # The third test verifies that the AA sequence begins with Met. 
        if single_aa_seq[0] == 'M':
                test[2] = 0

        # The fourth test verifies that the AA sequence ends with a stop codon (*).
        if single_aa_seq[-1] == 'Z':
                test[3] = 0

        # The fifth test verifies that the AA sequence only contains a single stop codon.
        if single_aa_seq.count('Z') == 1:
                test[4] = 0

        # The sixth test verifies that the AA sequence contains only the standard 20 AAs, plus stop (*).
        for aa in single_aa_seq:
                if aa not in residues:
                    test[5] = 1

        # The seventh test verifies that dna_len = 3*aa_len
        if len(sequence) == 3*len(single_aa_seq):
                test[6] = 0

    # The cumulative number of times each test failed is recorded and output during each iteration. 
    test_total = [test_total[i] + test[i] for i in range(len(test_total))]
    print(test_total)

    # If all 6 tests succeed, the DNA sequence and corresponding AA sequence are added to new lists. 
    if test == [0,0,0,0,0,0,0]:
        dna_seq_new.append(str(sequence))
        aa_seq_new.append(str(single_aa_seq))

# Shuffle items in the dictionary
seq_dict = {dna_seq_new[i]: aa_seq_new[i] for i in range (len(dna_seq_new))}
items = list(seq_dict.items())
random.shuffle(items)
dict_shuff = OrderedDict(items)

# Write to file
with open('cho_dict.json','w') as f:
        json.dump(dict_shuff, f)