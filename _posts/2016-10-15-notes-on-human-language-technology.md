---
layout:       post
title:        "Notes on Speech and Language Processing"
author:       "GP Huang"
---

A quick review of related concepts.

Table of Contents:

- [Language Modeling](#lm)
 - [Problem](#prob)
 - [Introduction and Intuition of N-Grams](#intro)
 - [NNLM](#nnlm)
 - [Benchmarking Test on English PTB](#bench)
- [Pronunciation Modeling]
- [Acoustic Modeling]
- [Corpus]
- [Rescoring]
- [Decoding]
- [Summary](#summary)
- [Additional references](#add)

<a name='prob'></a>

## Problems

we have sparse data

<a name='lm'></a>

## Language Modeling

Looking at the research topic, basic definitions, major problems, trend, and possible solutions.

<a name='intro'></a>

### Introduction and Intuition

**_Definition_**: the language model or LM, even better 'grammar' tell
- the probability of how likely is a sentence or sequence of words. $$P(w1,w2,\cdots,w_{n})$$
- the probability of an upcoming word: $$P(w_n|w1,w2,\cdots,w_{n-1})$$

Applications: machine translation, spell correction, question-answering, and speech recognition etc.

Computation: chain rule
- compute joint probability using conditional probability
- Immediate approach counts and divide: not realistic given usually limited amount of text data

$$
P(w1,w2,\cdots,w_{n})
= P(w_1)P(w_2|w_1)P(w_3|w1,w2) \cdots P(w_n|w_1,w_2,\cdots,w_{n-1})
= \prod_iP(w_i|w_1,w_2,\cdots,w_{i-1})
$$

Assumption by Andrei Markov
- use only adjacent k word(s)
- unigram: no k; bigram: k=1; trigram: k=2 etc. (confused? k is the number of neighbours you have, uni/bi/tri refers to the total number of tokens/grams)
$$
P(w_i|w_1,w_2,\cdots,w_{i-1}) \approx  = P(w_i|w_{i-k},\cdots,w_{i-1})
thus,
P(w1,w2,\cdots,w_{n}) \approx  = \prod_iP(w_i|w_{i-k},\cdots,w_{i-1})
$$
- maximum likelihood estimation, with bigram example:
$$
P(w_i|w_{i-1}) = \frac{count(w_{i-1},w_i)}{count(w_{i-1})}
$$

```unix
# txtf is some ascii or utf-8 encoded (English) text file
# tokinize by word
tr -sc '[A-Za-z]' '[\012*]' < txtf > txtf.words
tail +2 txtf.words > txtf.nextwords
paste txtf.words txtf.nextwords | sort | uniq -c | sort -nr > txtf.bigram

```

About N-grams
- N-grams models are insufficient models of languages, because language has **long-distance dependencies**
- The best LM is one that predicts an unseen test set with the highest P(sentence), thus lowest PPL

Solutions
- N-grams only work well for word prediction if the test corpus looks like the training corpus, thus we need to train robust LMs that generalize!
- Steal probability mass to generalize better, Add-1 estimate


PPL:
- logarithm: multiplication, addition
- Extrinsic evaluation: comparing models A and B in an actual task, e.g., rate of OOV reduction - time consuming
- Intrinsic evaluation: perplexity - bad approximation, subject to the matching condition of training/testing data, good for pilot experiments

$$
PPL = P(sentence) = P(w_1,w_2,\cdots,w_N)^{1/N}
$$

**_Generalization methods_**
- smoothing
 - add-1 (Laplace): blunt for N-gram, but decent to smooth other NLP models, e.g. text classification
 - add-k
 - intuitions: Good-Turing, Kneser-Ney, Witten-Bell
- backoff: use less context - use trigram if good evidence, otherwise bigram, otherwise unigram
- interpolation: mix unigram, bigram, trigram
 - linear
 - non-linear
- pruning: select N-grams
 - threshold-based, count > threshold
 - entropy-based
- others
 - discriminative models
 - parsing-based models
 - caching models
 
### Difficulties
Ok, it seems all you need is some text to build a LM, i.e. with count priors. But ALAS, the problem starts here, how do you handle/represent words? more specifically,

- where do you get the text data? 
- how much text data do you get?
- how reliable is the distribution of (word) tokens in the text data?
- are there any shorcomings of the LM?
- are there enough training examples for the words _seen_?
- for _unseen_ words, can try _part of speech abstraction_, _prefixes/suffixes abstraction_

These questions relate to the quality and quantity of data ressources that are often at the root of the problem.
That is to assume you have acquired at lease certain amount of wealth and human labor to collect the text data, or the scientific community at large have created a pool of shared resources, e.g. speech and text corpus, n-grams on certain languages.
Luckily, there are many such resources available, thanks to the recent popularity of Big Data. 
For example, you can have a look at [Google's Ngram Viewer](https://books.google.com/ngrams). Type in a word, and see how its usage has varied in history! 

Here are the two main difficulties, data sparsity and OOV, arising from the text data, they are somehow co-related, where the latter is much more specific and is often a direct threat for speech recognition and language understanding tasks.

**_Sparse Data_**

Sparsity and density


**_OOV Problem_**

induce word representations from unlabeled text

- HMM-based representations
- NN-based representations

 
### Current Trends

<a name='nnlm'></a>

### NNLM

The NN learns the embeddings of words, where all words are represented with D-dimensional vectors

- less data sparsity
- OOV words have non-zero probabilites

<a name='bench'></a>

### Benchmarking Test on English PTB

## Further Reading

  * [Data Sparsity in NLP](http://cogcomp.cs.illinois.edu/files/presentations/BGU_WordRepresentations_2009.pdf)

  * [Natural Language Processing, Prof. Jurafsky](https://class.coursera.org/nlp/lecture)

  * [Kneser-ney smoothing](http://www.foldl.me/2014/kneser-ney-smoothing/)
