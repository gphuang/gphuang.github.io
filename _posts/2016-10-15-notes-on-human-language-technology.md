---
layout:       post
title:        "Notes on Speech and Language Processing"
author:       "GP Huang"
---

A quick review of related concepts.

Table of Contents:

- [Language Modeling](#lm)
 - [Introduction and Intuiation of N-Grams]](#intro)
 - [NNLM](#nnlm)
 - [Benchmarking Test on English PTB](#bench)
- [Pronunciation Modeling]
- [Acoustic Modeling]
- [Corpus]
- [Rescoring]
- [Decoding]
- [Summary](#summary)
- [Additional references](#add)

<a name='lm'></a>
## Language Modeling

Looking at the research topic, basice definitions, major problems, trend, and possible solutions.

<a name='intro'></a>

### Introduction and Intuiation

**_Definition_**: the language model or LM, even better 'grammar' tell 
- the probablity of how likely is a sentence or sequence of words. $$P(w1,w2,\cdots,w_{n})$$
- the probablity of an upcoming word: $$P(w_n|w1,w2,\cdots,w_{n-1})$$

Applications: machine translation, spell correction, question-ansering, and speech recognition etc.

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
- bigram example:
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

N-grams models are insufficient models of languages, because language has **long-distance dependencies"

```

```

## Further Reading

  * [Natural Language Processing, Prof. Jurafsky](https://class.coursera.org/nlp/lecture)

  * [Kneser-ney smoothing](http://www.foldl.me/2014/kneser-ney-smoothing/)
