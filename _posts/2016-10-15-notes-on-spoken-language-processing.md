---
layout:       post
title:        "Notes on Spoken Language Processing"
author:       "GP Huang"
date:   2016-11-19 23:00:00
---

A quick review of related concepts.

Table of Contents:

- [Spoken Language Processing]
 - [ASR]
 - [NLP]
 - [TTS]
- [Language Modeling](#lm)
 - [Problem](#prob)
 - [Introduction and Intuition of N-Grams](#intro)
 - [NNLM](#nnlm)
 - [Benchmarking Test on English PTB](#bench)
- [Pronunciation Modeling]
- [Acoustic Modeling]
- [Decoder]
- [Evaluation]
 - [Corpus]
 - [Rescoring]
- [Decoding LM+AM: Search Method](#decoder)
- [End-to-End System](#end2end)
- [Further Reading](#reading)

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
- are there any shortcomings of the LM?
- are there enough training examples for the words _seen_?
- for _unseen_ words, can try _part of speech abstraction_, _prefixes/suffixes abstraction_

These questions relate to the quality and quantity of data resources that are often at the root of the problem.
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
- OOV words have non-zero probabilities

Document as vectors

- distance as a bad idea
- use angle instead of distance

term frequency weighting

<a name='bench'></a>

RNNLM

character rnn language modelling, sample text from initial index/character

### Benchmarking Test on English PTB

<a name='decoder'></a>

## Decoding LM+AM: Search Method

source: http://anie.me/rnn-encoder-decoder/

RNN encoder-decoder has become the next heated field in Natural Language Processing, here is a history of its development. Papers are organized in a timely manner.

Ian Goodfellow’s book has a great overall summary to it: http://www.deeplearningbook.org/contents/rnn.html but sadly since it is a book, it lacks any recent development and any technical details.

Another interesting, more folksy way to approach is from WildML’s recent post: http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/

RNN encoder-decoder network was started by Kyunghyun Cho (Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation) and Ilya Sutskever (Sequence to Sequence Learning with Neural Networks) both in 2014.

Sutskever’s model constitutes end-to-end (sequence-to-sequence) training (with pure RNN), while Cho’s model relies on scoring proposals generated by another machien translation system.

A huge drawback of these two initial models are they both use a fixed-length vector to represent the meaning of a source sentence. The dimension of this fixed-length vector can get too small to summarize a long sequence.

Bahdanau observed this phenomenon in his paper in 2015:, which became the second milestone paper for RNN encoder-decoder (NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE). He proposed a variable length vector C to summarize the context of the source sentence, and the most infamous attention mechanism. It’s interesting to notice that the so-called “attention” is used interchangeably as “alignment” in Bahdanau’s paper, because attention mechanism is analogous to alignment in traditional machien translation literature.

This brings us to another paper, also published in 2015, that sheds light on global vs. local attention (i.e., soft vs. hard attention in Computer Vision), in Luong’s paper (Effective Approaches to Attention-based Neural Machine Translation). Local attention is inferred using a Gaussian distribution.

At last, let’s mention NLI: Natural Language Inference. With the SNLI corpus, NLI shares many common features as MT system, as pointed out in Bill McCartney’s paper: A Phrase-Based Alignment Model for Natural Language Inference, and one of them is alignment.

The RNN encoder-decoder scheme is first used in NLI by Rocktaschel in his paper: Reasoning about Entailment with Neural Attention in 2015, then Cheng proposed a different model: shallow attention fusion vs. deep attention fusion in the paper Long Short-Term Memory-Networks for Machine Reading, with the main difference that shallow attention fusion only summarize attention information at the last time step of the decoder (obviously the wrong way), and deep fusion summarizes attention information at each step, and merge hidden state of the encoder into the decoder at each time step, and such merging process is a weighted sum based on how much attention given decoder time step has to each time step from the encoder

<a name='end2end'></a>

## End-to-End System

<a name='reading'></a>

## Further Reading

  * [Data Sparsity in NLP](http://cogcomp.cs.illinois.edu/files/presentations/BGU_WordRepresentations_2009.pdf)

  * [Natural Language Processing, Prof. Jurafsky](https://class.coursera.org/nlp/lecture)

  * [Kneser-ney smoothing](http://www.foldl.me/2014/kneser-ney-smoothing/)
