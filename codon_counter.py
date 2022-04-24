import os

os.chdir('/mnt/c/Users/dgoulet/Scripts/codon-optimization/')

with open('pdl1-gm.txt') as f:
    seq_1 = f.read().upper()

with open('pdl1-rnn.txt') as f:
    seq_2 = f.read().upper()

def encrypt(string,length):
    return ' '.join(string[i:i+length] for i in range(0,len(string),length))

codons_1_pre = encrypt(seq_1,3)
codons_2_pre = encrypt(seq_2,3)

codons_1 = codons_1_pre.split()
codons_2 = codons_2_pre.split()

seq_1_nospace = seq_1.replace(" ", "")
seq_2_nospace = seq_2.replace(" ", "")

bases_1 = [char for char in seq_1_nospace]
bases_2 = [char for char in seq_2_nospace]

yes = 0
no = 0

for i in range(len(codons_1)):
    if codons_1[i] == codons_2[i]:
        yes += 1
    else:
        no += 1

hai = 0
iie = 0

for j in range(len(bases_1)):
    if bases_1[j] == bases_2[j]:
        hai += 1
    else:
        iie += 1

total_overall = hai + iie
percent_overall = hai / total_overall * 100

total = yes + no
percent_match = yes / total * 100

print(f'The DNA sequences are {round(percent_overall,1)}% identical. ({hai}/{total_overall} bases)')
print(f'The codon usage is {round(percent_match,1)}% identical. ({yes}/{total} codons)')

seq1_a = 0
seq1_c = 0
seq1_g = 0
seq1_t = 0

for base in bases_1:
    if base == 'A':
        seq1_a += 1
    elif base == 'C':
        seq1_c += 1
    elif base == 'G':
        seq1_g += 1
    elif base == 'T':
        seq1_t += 1

seq1_total = seq1_a + seq1_c + seq1_g + seq1_t
seq1_pct_a = round(seq1_a / seq1_total * 100, 1)
seq1_pct_c = round(seq1_c / seq1_total * 100, 1)
seq1_pct_g = round(seq1_g / seq1_total * 100, 1)
seq1_pct_t = round(seq1_t / seq1_total * 100, 1)

print('Seq1:')
print(f'A: {seq1_pct_a}')
print(f'C: {seq1_pct_c}')
print(f'G: {seq1_pct_g}')
print(f'T: {seq1_pct_t}')

seq2_a = 0
seq2_c = 0
seq2_g = 0
seq2_t = 0

for base in bases_2:
    if base == 'A':
        seq2_a += 1
    elif base == 'C':
        seq2_c += 1
    elif base == 'G':
        seq2_g += 1
    elif base == 'T':
        seq2_t += 1

seq2_total = seq2_a + seq2_c + seq2_g + seq2_t
seq2_pct_a = round(seq2_a / seq2_total * 100, 1)
seq2_pct_c = round(seq2_c / seq2_total * 100, 1)
seq2_pct_g = round(seq2_g / seq2_total * 100, 1)
seq2_pct_t = round(seq2_t / seq2_total * 100, 1)

print('Seq2:')
print(f'A: {seq2_pct_a}')
print(f'C: {seq2_pct_c}')
print(f'G: {seq2_pct_g}')
print(f'T: {seq2_pct_t}')
