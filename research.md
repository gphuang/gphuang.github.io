---
layout: page
permalink: /research/
---

My research experience started with a project during my final year at University, to design a software system to help student to speak standard English.
From there it has been an enriching journey in human language technologies, especially on pronunciation modeling and language modeling.
The purpose of this page is to outline the main topics on which I publish my research results.

## Publications

**An Investigation into LM Data Augmentation for Low-resourced ASR And KWS**

[[pdf]](/papers/2017_ICASSP_LIMSI/huang2017investigation.pdf)  [[bib]](/papers/2017_ICASSP_LIMSI/huang2017investigation.bib) ICASSP 2017

<div class="imgcap">
<div>
<img src="/images/huang2017investigation.png" style="max-width:60%;  text-align:justify;">
</div>
<div class="thecap">Figure: Evaluate LM perplexity on Babel low-resourced languages.
</div>
</div>

The linguistic regularities in the text data can be captured by recurrent neural networks (RNNs), which represent words by vectors, i.e., the input-layer weights.
Taking advantage of the continuous-space word vectors, RNN based Language Models (LMs) have gained much momentum in the past few years.
Compared with n-grams, the RNN-LMs provide better vocabulary coverage and reduce data sparsity.
This is especially useful when dealing with out-of-vocabulary (OOV) words in ASR systems.

I train various RNN-LMs using word and subword units, which then generate artificial text data to optimize the performance of LMs for ASR and keyword search (KWS) tasks.
Word-based RNNs keep the same vocabulary so do not impact the OOV, whereas subword units can reduce the OOV rate.
In particular, LMs including text data generated with RNNLMs improve ASR and KWS results on several low-resourced languages, such as Cantonese, Mongolian, Igbo, and Amharic.

More on

  * [Academic Page]() (note: I will try to upload the preprints to my new webpage, and the scripts to github in the near future.)

  * [Google Scholar](https://scholar.google.fr/citations?user=hrICCP0AAAAJ&hl=en)

## Experiences

**2016 - present** Post-doc: Acoustic Events Analysis

**2015 - 2016** Post-doc: Babel - Text Data Augmentation via Statistical Machine Translation and RNNLMs (Cantonese, English)

**2012 - 2015** Post-doc: Speech Segmentation with Multi-View Features (Malay, English)

**2008 - 2012** Doctoral: Automatic Speech Recognition with Articulatory Phonetic Features (English)

**2004 - 2008** Bachelor: Digital Signal Processing: Audio/Image/Video; Information Theory; Literature Studies

## Teaching

#### Spring/Fall 2009-2010

I was a teaching assistant for Engineering Mathematics.
This was an undergrad-level course intended to introduce the basics of Mathematics to 2nd year University students.
