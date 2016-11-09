---
layout:       post
title:        "Notes on Linear Algebra"
---

## Matrix basics

* Problem setting

```
1x + 2y + 3 z = 12
2x + 3y + 4z = 11
3x + 4y + 5z = 10
```

* Formulate with numpy

```python
A*X = b
A = np.array(np.mat(‘1 2 3; 2 3 4; 3 4 5’))
A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
X = np.transpose(np.array([x, y, z]))
```

* Rule of thumb

```
row x matrix = row\_picture
matrix x column = col\_picture
row x column = scalar
elimination views matrix as stacked rows or cols
permutation views manipulate row/col vectors
```

## Generate a spiral dataset and plot

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
  


