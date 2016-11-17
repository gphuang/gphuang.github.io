---
layout: page
permalink: /research/
---

## Publications

### RNNLM for Text Data Augmentation

The linguistic regularities in the text data can be captured by recurrent neural networks (RNNs), which represent words by vectors, i.e., the input-layer weights.
Taking advantage of the continuous-space word vectors, RNN based Language Models (LMs) have gained much momentum in the past few years.
Compared with n-grams, the RNN-LMs provide better vocabulary coverage and reduce data sparsity.
This is especially useful when dealing with out-of-vocabulary (OOV) words in ASR systems.

I train various RNN-LMs using word and subword units: characters and morphs, which then generate artificial text data to optimize the performance of LMs for ASR and keyword search (KWS) tasks.
Word-based RNNs keep the same vocabulary so do not impact the OOV, whereas subword units can reduce the OOV rate.
In particular, LMs including text data generated with RNNLMs improve ASR and KWS results on several low-resourced languages, such as Cantonese, Mongolian, Igbo, and Amharic.

[pdf] [BibTex] ICASSP'16  INTERSPEECH'16

### Articulatory Features for Speech Recognition

Acoustic-to-articulatory mapping is a challenging topic in speech research.
Due to the nature of recording conditions, the articulatory data are less abundant than the acoustic data, thus they are usually retrieved or inverted from the latter.
The main difficulty is that there exist multiple articulatory configurations that could produce the same acoustic output, i.e., the “many to one” problem.

To alleviate the “many to one” problem, I hypothesize that the muscular control signals are correlated with the articulatory trajectories and the acoustic features of human speech.
I use the muscular control signal to constrain the number of articulatory configurations.
I apply DNN to predict the articulatory trajectories from the acoustic features, e.g., MFCCs.
Using a 2-hidden layer DNN, I obtain an average root mean square error below 1 mm on the EMA measures of selected articulator positions: upper/lower lips, lower incisor, tongue tip, tongue body, and tongue dorsum, on the multi-channel articulatory corpus, MOCHA-TIMIT.
The retrieved articulatory trajectories and the acoustic features are then used together to improve the ASR performance on TIMIT data.

[pdf] [BibTex] Computer Speech and Language Volume 28, pp. 163-176, 2014.

### TODO

[Google Scholar Page](Google)

[Project](http) [PDF](http) [Code (Github)](http)

## Teaching

### Spring/Fall 2009-2010

I was a teaching assistant for Engineering Mathematics.
This was an undergrad-level course intended to introduce the basics of Mathematics to 2nd year University students.
