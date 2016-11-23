---
layout:       post
title:        "Notes on Linear Algebra"
author:       "GP Huang"
---

A quick review of related concepts.

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

## Optimization

**_gradient descent_** mini-batch GD, stochastic GD (i.e. on-line gradient descent)

## Backpropagation

computing the gradient analytically using the chain rule, i.e. backpropagation.

Intuition with gates

* mini-batch SGD loop:
  - sample
  - forward loss
  - backward gradient
  - update weights

* layer - gate
* weight - parameters
* forward - backward pass
