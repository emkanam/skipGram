# Introduction

This projet is from exercise 1 of the NLP course at MVA and CentraleSupélec.

# Our Model

The skipGram model learns two representations: `context` and `target`.

the loss was exploding so we used a modified sigmoid function to make sure 

we created a function executionTime that we use as decorator to watch the execution time of our functions to optimize the process.

we tried doc = nlp(l.lower())
no_punc_sentence = [token.orth_ for token in doc if not token.is_punct | token.is_space]
but too long

so we used string translation went from 6s to 0.12s -> loading 100.000 sentences in 6s

# Results