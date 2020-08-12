with open('seq-opt.txt') as f:
    seq_idt = f.read().upper()

with open('seq-orig.txt') as f:
    seq_opt = f.read().upper()

def encrypt(string,length):
    return ' '.join(string[i:i+length] for i in range(0,len(string),length))

codons_opt_pre = encrypt(seq_opt,3)

codons_idt = seq_idt.split()
codons_opt = codons_opt_pre.split()

seq_idt_nospace = seq_idt.replace(" ", "")
seq_opt_nospace = seq_opt.replace(" ", "")

bases_idt = [char for char in seq_idt_nospace]
bases_opt = [char for char in seq_opt_nospace]

yes = 0
no = 0

for i in range(len(codons_idt)):
    if codons_idt[i] == codons_opt[i]:
        yes += 1
    else:
        no += 1

hai = 0
iie = 0

for i in range(len(bases_idt)):
    if bases_idt[i] == bases_opt[i]:
        hai += 1
    else:
        iie += 1

total_overall = hai + iie
percent_overall = hai / total_overall * 100

total = yes + no
percent_match = yes / total * 100

print(f'The DNA sequences are {round(percent_overall,1)}% identical. ({hai}/{total_overall} bases)')
print(f'The codon usage is {round(percent_match,1)}% identical. ({yes}/{total} codons)')

idt_a = 0
idt_c = 0
idt_g = 0
idt_t = 0

for base in bases_idt:
    if base == 'A':
        idt_a += 1
    elif base == 'C':
        idt_c += 1
    elif base == 'G':
        idt_g += 1
    elif base == 'T':
        idt_t += 1

idt_total = idt_a + idt_c + idt_g + idt_t
idt_pct_a = round(idt_a / idt_total * 100, 1)
idt_pct_c = round(idt_c / idt_total * 100, 1)
idt_pct_g = round(idt_g / idt_total * 100, 1)
idt_pct_t = round(idt_t / idt_total * 100, 1)

print('IDT:')
print(f'A: {idt_pct_a}')
print(f'C: {idt_pct_c}')
print(f'G: {idt_pct_g}')
print(f'T: {idt_pct_t}')

opt_a = 0
opt_c = 0
opt_g = 0
opt_t = 0

for base in bases_opt:
    if base == 'A':
        opt_a += 1
    elif base == 'C':
        opt_c += 1
    elif base == 'G':
        opt_g += 1
    elif base == 'T':
        opt_t += 1

opt_total = opt_a + opt_c + opt_g + opt_t
opt_pct_a = round(opt_a / opt_total * 100, 1)
opt_pct_c = round(opt_c / opt_total * 100, 1)
opt_pct_g = round(opt_g / opt_total * 100, 1)
opt_pct_t = round(opt_t / opt_total * 100, 1)

print('Opt:')
print(f'A: {opt_pct_a}')
print(f'C: {opt_pct_c}')
print(f'G: {opt_pct_g}')
print(f'T: {opt_pct_t}')

