---
title:        "Notes on Machine Learning with NN"
# jekyll-seo-tag
description:  "As title indicates"
author:       "GP Huang"
---

# ML Basics
The fundamental goal of machine learning is to generalize beyond the examples in the training set.

Clustering is a type of unsupervised learning: input data samples have no output labels

[K-Nearest Neighbor (KNN)](http://cs231n.github.io/classification/)

K-NN’s success is greatly dependent on the representation it classifies data from, so one needs a good representation before k-NN can work well.
They are very expensive to train, but once the training is finished it is very cheap to classify a new test example. This mode of operation is much more desirable in practice.

[Linear Classification]( http://cs231n.github.io/linear-classify/)

Problem setting

```
1x + 2y + 3 z = 12

2x + 3y + 4z = 11

3x + 4y + 5z = 10
```

Formulate with numpy

```python
A*X = b
A = np.array(np.mat(‘1 2 3; 2 3 4; 3 4 5’))
A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
X = np.transpose(np.array([x, y, z]))
```

Loss function

```python
def L_ivectorized(x, y, w):
    scores = w.dot(x)
    margins = np.maximum(0, scores - scores[y] +1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i
```
 
# DNN
## Questions
* what a neural network is really doing, behaviour of deep neural networks
* how it is doing so, when succeeds; and what went wrong when fails.
  - explore low-dimensional deep neural networks when classifying certain datasets
  - visualise the behavior, complexity, and topology of such networks
* On an example dataset, with each layer, the network transforms the data, creating a new representation

## Continuous Visualization of Layers
* A linear transformation by the “weight” matrix
* A translation by the vector
* Point-wise application of e.g., tanh.

## Backpropagation

* mini-batch SGD loop:
  - sample
  - forward loss
  - backward gradient
  - update weights 

* layer - gate
* weight - parameters
* forward - backward pass

# Application to Audio: Human Speech and Language Processing


# Application to Vision: Image and Video Processing

## Historical Context
* cat & edges, how brain neurons work, 1981, 1963
* 1966 A.I. Computer Vision


<p class="lead">Further Reading</p>

* [ML Course, Prof. Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome)

* [DNN Topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

* [Slides: A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
