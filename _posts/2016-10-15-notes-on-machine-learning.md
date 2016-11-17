---
layout:       post
mathjax:      true
title:        "Notes on Machine Learning with NN"
---

The fundamental goal of machine learning is to generalize beyond the examples in the training set.

## Goal: build a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier.

  * understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
  * understand the train/val/test splits and the use of validation data for hyperparameter tuning.
  * develop proficiency in writing efficient vectorized code with numpy
  * implement and apply a k-Nearest Neighbor (kNN) classifier
  * implement and apply a Multiclass Support Vector Machine (SVM) classifier
  * implement and apply a Softmax classifier
  * implement and apply a Two layer neural network classifier
  * understand the differences and tradeoffs between these classifiers
  * get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)


### [Nearest Neighbor Classifier](http://cs231n.github.io/classification/)

Clustering is a type of unsupervised learning: input data samples have no output labels. For example, K-Nearest Neighbor (KNN)

K-NN’s success is greatly dependent on the representation it classifies data from, so one needs a good representation before k-NN can work well.
They are very expensive to train, but once the training is finished it is very cheap to classify a new test example. This mode of operation is much more desirable in practice.

### [Linear Classifier]( http://cs231n.github.io/linear-classify/)
training dataset $$ (x_{i}, y_{i}) $$. $$ i = 1 \cdots N $$
N examples each with dimensionality D
K distinct categories

$$
f(x,y) = x y
$$

$$
f(x_i, W) = W*x_i
$$

<div>
$$
\frac{\partial f(x,y)}{\partial x} = \frac{f(x+h,y) - f(x,y)}{h}
$$
</div>

weights, parameters

Data preprocessing: mean subtraction, scale, [-1, 1]

### Loss function

cost function, objective function

### Support Vector Machines

### Softmax ?

### Linear Regression

$$ \begin{align}
X &= [1, X_1 X_2 \cdots X_n]
  &= \begin{bmatrix}
1 &x_1^{(1)}  &\cdots  & x_n^{(1)} \\
1 &x_1^{(2)}  &\cdots  &x_n^{(2)} \\
\vdots  &\vdots  &\ddots   &\vdots \\
1 &x_1^{(m)}  &\cdots  &x_n^{(m)}
\end{bmatrix} \\

\theta &=\begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \\ \end{bmatrix}
\end{align} $$

 i.e., $$ X_0 = 1 $$


## Goal: writing backpropagation code, and training Neural Networks and Convolutional Neural Networks.

  * understand Neural Networks and how they are arranged in layered architectures
  * understand and be able to implement (vectorized) backpropagation
  * implement various update rules used to optimize Neural Networks
  * implement batch normalization for training deep networks
  * implement dropout to regularize networks
  * effectively cross-validate and find the best hyperparameters for Neural Network architecture

### Optimisation: Stochastic Gradient Descent

### Backpropagation
#### Intuition with gates

* mini-batch SGD loop:
  - sample
  - forward loss
  - backward gradient
  - update weights

* layer - gate
* weight - parameters
* forward - backward pass

### [Neural Classifier](http://cs231n.github.io/neural-networks-1/)
### #Questions
* what a neural network is really doing, behaviour of deep neural networks
* how it is doing so, when succeeds; and what went wrong when fails.
  - explore low-dimensional deep neural networks when classifying certain datasets
  - visualise the behavior, complexity, and topology of such networks
* On an example dataset, with each layer, the network transforms the data, creating a new representation

### Continuous Visualization of Layers
* A linear transformation by the “weight” matrix
* A translation by the vector
* Point-wise application of e.g., tanh.

## Goal: implement recurrent networks, and apply them to image captioning on Microsoft COCO.

  * Understand the architecture of recurrent neural networks (RNNs) and how they operate on sequences by sharing weights over time
  * Understand the difference between vanilla RNNs and Long-Short Term Memory (LSTM) RNNs
  * Understand how to sample from an RNN at test-time
  * Understand how to combine convolutional neural nets and recurrent nets to implement an image captioning system
  * Understand how a trained convolutional network can be used to compute gradients with respect to the input image
  * Implement and different applications of image gradients, including saliency maps, fooling images, class visualizations, feature inversion, and DeepDream.


## Case Studies
### Problem setting 1: linear dataset

```
1x + 2y + 3 z = 12

2x + 3y + 4z = 11

3x + 4y + 5z = 10
```

Formulate with numpy

```python
import numpy as np

A*X = b
A = np.array(np.mat(‘1 2 3; 2 3 4; 3 4 5’))
A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
X = np.transpose(np.array([x, y, z]))
# visualize the data
TBD
```

### Problem setting 2: non-linear (spiral) dataset

```python
import numpy as np

N = 100 # number of points per class/spiral wing
D = 2  # dimensionality
K = 5  # number of spiral wings
X = np.zeros((N*K)) # data matrix (each row is a single training example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in xrange(K):
  ix = range(N*j, N*(j+1))
  r = np.linspace(0.0, 1, N) # radius
  t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# visualize the data
plt.scatter(X[:,0],X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
```
## Loss function

```python
def L_ivectorized(x, y, w):
    scores = w.dot(x)
    margins = np.maximum(0, scores - scores[y] +1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i
```

## Application to Audio: Human Speech and Language Processing

## Application to Vision: Image and Video Processing

### Historical Context
* cat & edges, how brain neurons work, 1981, 1963
* 1966 A.I. Computer Vision


<p class="lead">Further Reading</p>

* [ML Course, Prof. Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome)

* [DNN Topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

* [Slides: A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
