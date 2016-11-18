---
layout: page
permalink: /research/
---


## Motivations
  * to analyze speech/sounds with respect to articulatory and auditory cues.
  * to imitate the dual-process of human speech production and perception.
  * to develop intelligent computer systems capable of understanding natural language.

## Experiences

**2016 - present**

**2015 - 2016** Post-doc: Babel Project - Text Data Augmentation (Cantonese, English)

**2012 - 2015** Post-doc: Speech Segmentation with Multi-View Features (Malay, English)

**2008 - 2012** Doctoral: Automatic Speech Recognition with Articulatory Phonetic Features (English)

**2004 - 2008** Digital Signal Processing: Audio/Image/Video; Information Theory; Literature Studies

## Publications

### Text Data Augmentation for Babel Under-Resourced Languages

The linguistic regularities in the text data can be captured by recurrent neural networks (RNNs), which represent words by vectors, i.e., the input-layer weights.
Taking advantage of the continuous-space word vectors, RNN based Language Models (LMs) have gained much momentum in the past few years.
Compared with n-grams, the RNN-LMs provide better vocabulary coverage and reduce data sparsity.
This is especially useful when dealing with out-of-vocabulary (OOV) words in ASR systems.

I train various RNN-LMs using word and subword units, which then generate artificial text data to optimize the performance of LMs for ASR and keyword search (KWS) tasks.
Word-based RNNs keep the same vocabulary so do not impact the OOV, whereas subword units can reduce the OOV rate.
In particular, LMs including text data generated with RNNLMs improve ASR and KWS results on several low-resourced languages, such as Cantonese, Mongolian, Igbo, and Amharic.

[PDF](link to host) [BibTex]

### Machine Translation Based Data Augmentation For Cantonese Keyword Spotting

This paper presents a method to improve a language model for a limited-resourced language using statistical machine translation from a related language to generate data for the target language. In this work, the machine translation model is trained on a corpus of parallel Mandarin-Cantonese subtitles and used to translate a large set of Mandarin conversational telephone transcripts to Cantonese, which has limited resources. The translated transcripts are used to train a more robust language model for speech recognition  and  for keyword  search  in  Cantonese  conversational  telephone  speech.This  method  enables  the  keyword  search  system  to  detect 1.5 times more out-of-vocabulary words,  and achieve 1.7% absolute improvement on actual term-weighted value.

[PDF](link to host) [BibTex] ICASSP 2016

### Articulatory Phonetic Features for Speech Recognition

Acoustic-to-articulatory mapping is a challenging topic in speech research.
Due to the nature of recording conditions, the articulatory data are less abundant than the acoustic data, thus they are usually retrieved or inverted from the latter.
The main difficulty is that there exist multiple articulatory configurations that could produce the same acoustic output, i.e., the “many to one” problem.

To alleviate the “many to one” problem, I hypothesize that the muscular control signals are correlated with the articulatory trajectories and the acoustic features of human speech.
I use the muscular control signal to constrain the number of articulatory configurations.
I apply DNN to predict the articulatory trajectories from the acoustic features, e.g., MFCCs.
Using a 2-hidden layer DNN, I obtain an average root mean square error below 1 mm on the EMA measures of selected articulator positions: upper/lower lips, lower incisor, tongue tip, tongue body, and tongue dorsum, on the multi-channel articulatory corpus, MOCHA-TIMIT.
The retrieved articulatory trajectories and the acoustic features are then used together at the phonetic level to improve the ASR performance on TIMIT data.

[pdf](link to host) [BibText] Thesis 2013

[pdf](link to host) [BibTex] Computer Speech and Language Volume 28, pp. 163-176, 2014.

[pdf](link to host) [BibText] ASRU 2011

[pdf](link to host) [BibText] IJCNN 2010


### TODO

[Google Scholar Page](Google)

[PDF](http) [BibText] [Code (Github)]

## Teaching

### Spring/Fall 2009-2010

I was a teaching assistant for Engineering Mathematics.
This was an undergrad-level course intended to introduce the basics of Mathematics to 2nd year University students.
