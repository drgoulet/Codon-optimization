# Codon-optimization
Use machine learning to design codon-optimized DNA sequences for increased protein expression in CHO cells.

To predict the 'best' DNA sequence for a given amino acid sequence using the pre-trained model, use rnn_predict.py.

Packages required:
- Python (built using Python 3.7.4 via Ubuntu-18.04)
- Keras (tensorflow)
- json
- NumPy
- OS

Codon optimization procedure:
1) Place the following in the same directory:
- rnn_predict.py
- rnn_model.h5
- dna_tokenizer.json
- aa_tokenizer.json
- Text file containing amino acid sequence (save as "sequence.txt")
  - Single-letter amino acid abbreviations, no spaces
  - Signal peptide included, if applicable
  - No stop codon (will be added to output DNA sequence)

2) In line 19 of rnn_predict.py, change to the working directory containing the above files.
3) Run rnn_predict.py.
4) The optimized DNA sequence is output as "sequence_opt.txt" in the same directory. 

