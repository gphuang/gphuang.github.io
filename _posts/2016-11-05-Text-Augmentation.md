---
layout:       post
title:        "Text Data Augmentation"
---

<p class="lead">Question, could you get more data from the available text data, i.e., text data augmentation?</p>

## Introduction

This is a continuously going research work.

**Motivation.** The Text Augmentation problem, which is the task of generating additional text data from any existing set of text data, usually from some under-resourced languages, i.e., less than 100 hours of speech or less than 100K words in the training transcripts. (Babel Project)
This is one of the key areas in Human Language Technology (HLT) that has a many practical applications, such as speech recognition, machine translation, and language modeling.
If HLT is to propagate to any new/old languages, including 女书, Elvish, and Klingon, it will indefinitely benefit from text augmentation.

**Example.** For example, in the image below are some texts extracted from LOTR, written in Elvish with English translations.

INSERT image

**Challenges.** Since collecting millions of words from an under-resourced language is relatively difficult and usually time/labor/money consuming, it is worth considering the challenge from the perspective of an automated text generation algorithm, i.e., using existing text to make more text

  * find out the word composition, the lexicon, the syntax, the semantics, and other linguistic regularities.
  * danger of overfitting, if sample is too small/redundant
  * different languages have rather diverse regularities/rules

A good text augmentation model must be invariant to the above variations, while retaining the ability to adapt to different language (groups).

How do we go about writing an algorithm that can discover, generate, and build language models?

**Data-driven**
(almost) purely based on decomposition, parallelism.

"""
the approach that we will take is not unlike one you would take with a child: we’re going to provide the computer with many examples of each class and then develop learning algorithms that look at these examples and learn about the visual appearance of each class. This approach is referred to as a data-driven approach, since it relies on first accumulating a training dataset of labeled images. Here is an example of what such a dataset might look like:
"""

**Knowledge-based**
Working with linguists, with computational approach in mind. JR Meyers

**The pipline**

  * Input:
  * Learning:
  * Evaluation:

## TG1

## TG2

## Related works (Literature review):
- Recurrent Neural Networks

Basics:
- transformation to expand training set
- e.g., image horizontal flips, crops/scales, color jitter
- simple to implement and use
- useful for small datasets

## Summary

In summary:

  * introduce to the problem of  **Text Augmentation**
  * introduce to a simple algorithm
  * build a pipeline
  * show
