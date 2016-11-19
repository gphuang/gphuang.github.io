---
layout:       post
title:        "Notes on Machine Learning with NN"
author:       "GP Huang"
date:   2016-11-19 22:00:00
mathjax: true
---

The fundamental goal of machine learning is to generalize beyond the examples in the training set.

## Goal:
  * build a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier.
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

**_1 Nearest Neighbor classifier_**
</div>
```python
import numpy as np

# input
Xtr, Ytr, Xte, Yte = loadData('path') # magic function
# training
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr, Ytr) # train the classifier on the training data and labels
# testing
Yte_predict = nn.predict(Xte) # predict labels on the test data
# print the classification accuracy: average number of examples that are correctly predicted, label matches
print 'accuracy: %f' % (np.mean(Yte_predict == Yte))
# the classifier
class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is N x 1 """
    # the nearest neighbor classifier simply remembers ALL the training data
    slef.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example, which we wish to predict label for """
    num_test = X.shape[0]
    # make sure the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      ## or using the L2 distance
      # distance = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
      min_index = np.argmin(distances) # get the index with the smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
    return Ypred
```
</div>

**_K Nearest Neighbor classifier_**
</div>
```python
Xval = Xtr[:1000, :] # take the first 1000 for validation
Yval = Ytr[:1000]
Xtr = Xtr[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find Hyperparameters that work best on the validatio set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
  # use a particular value of k and evaluate on validation data
  nn = NearestNeighbor()
  nn.train(Xtr, Ytr)
  # assume a modified NearestNeighbor class that can take k as input
  Yval_predict = nn.predict(Xval, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```
</div>


### [Linear Classifier]( http://cs231n.github.io/linear-classify/)

**_Disadvantages_** of

#### Score Function
**_Score Function_**

#### Loss Function

**_loss function_** measures our unhappiness with outcomes, sometimes also referred to as the cost function or the objective function.

**_Multiclass SVM_**

$$
L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) \right] + \lambda \sum_k\sum_l W_{k,l}^2
$$

**_Softmax_**
**_Compare_**

$ \sum_{\forall i}{x_i^{2}} $

\\(P(y = 1 \mid x; w,b) = \sigma(w^Tx + b)\\), where \\(\sigma(z) = 1/(1+e^{-z})\\)

training dataset \\(x\_i\\), y_{i})\\), where $ i = 1 \cdots N $


N examples each with dimensionality D
K distinct categories

$$
f(x,y) = x y
$$

$$
f(x_i, W) = W*x_i
$$

$$
\frac{\partial f(x,y)}{\partial x} = \frac{f(x+h,y) - f(x,y)}{h}
$$

weights, parameters

Data preprocessing: mean subtraction, scale, [-1, 1]

Validation sets for Hyperparameter tuning

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
