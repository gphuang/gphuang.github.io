---
layout:       page
title:        "Notes on Linear Algebra"
author:       "GP Huang"
---

A quick review of related concepts.

## Matrix basics

### Problem setting

```
1x + 2y + 3 z = 12
2x + 3y + 4z = 11
3x + 4y + 5z = 10
```

### Formulate with numpy

```python
A*X = b
A = np.array(np.mat(‘1 2 3; 2 3 4; 3 4 5’))
A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
X = np.transpose(np.array([x, y, z]))
```

### Vector, Matrix, and Tensor Derivatives

more at http://cs231n.stanford.edu/vecDerivs.pdf


### Rule of thumb

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

dfdx, or dx, is the gradient on x, the partial derivative on x

the gradient is the vector [dfdx, dfdy, dfdz] or [dx, dy, dz]

```python
# set inputs
x = -2; y = 5; z = -4

# perform the forward pass
q = x + y # q becomes 3
f = q * z   # f becomes -12

# perform the backward pass (backpropagation, reverse-mode differentiation)
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4
# now backprop through q = x + y
dfdx = 1.0*dfdq # df/dx = dq/dx * df/dq = 1*df/dq
dfdy = 1.0*dfdq # df/dy = dq/dy * df/dq = 1*df/dq
```

```python
# assume some random weights and data 
w = [2,-3,-3] 
x = [-1,-2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2] # dot product x.dot(w.T)
f = 1.0/(1+math.exp(-dot)) # sigmoid function 

# backward pass through the neuron
ddot = (1-f)*f # sigmoid gradient derivation 
dx = [w[0] * ddot, w[1] * ddot] # bp into x, df/dx = df/ddot * ddot/dx = df/ddot * w
dw = [x[0] * ddot, x[1] * ddot, 1.0*ddot ] # bp into w df/dw = df/ddot * ddot/dw = df/ddot * x

```

In practice: staged computation

$$
f(x, y) = \frac{x + \sigma(y)}{\sigma(x) + (x+y)^2}
$$

To differentiate w.r.t. x or y is very complex.
But we do not need to have an explicit function to evaluate the gradient.
We only need to know how to compute it.

```python
# some inputs
x = 3
y = -4

# forward pass
sigy = 1.0/(1 + math.exp(-y)) # sigmoid in numerator   #(1)
num = x + sigy # numerator                             #(2)
sigx = 1.0/(1 + math.exp(-x)) # sigmoid in denominator #(3)
xpy = x + y                                            #(4) 
xpysqr = xpy**2                                        #(5)
den = sigx + xpysqr # denominator                      #(6) 
invden = 1.0/den                                       #(7)
f = num * invden                                       #(8)

# backward pass
# bp f = num * invden
dnum = invden                         #(8)
dinvden = num                         #(8)
# bp invden = 1.0/den 
dden = (-1.0/(den**2)) * dinvden      #(7)
# bp den = sigx + xpysqr 
dsigx = (1.0)*dden                    #(6)
dxpysqr = (1.0)*dden                  #(6)
# bp xpysqr = xpy**2                  
dxpy = (2.0*xpy)*dxpysqr              #(5)
# bp xpy = x + y                       
dx = 1.0*dxpy                         #(4)
dy = 1.0*dxpy                         #(4)
# bp sigx = 1.0/(1 + math.exp(-x))
dx += sigx * (1 - sigx)*dsigx         #(3)
# bp num = x + sigy 
dx += (1.0)*dnum                      #(2)
dsigy = (1.0)*dnum                    #(2)
# bp sigy = 1.0/(1 + math.exp(-y))
dy += sigy * (1 - sigy)*dsigy         #(1)

```

Tips:
    * Cache forward pass variables.
    * Gradients add up at forks. "+=" multi-variable chain rule, if a variable branches out at different parts of the circuit, then the gradients that flow back to it will add up.

Gradients for vectorized operations

**_Matrix-Matrix_** multiply graident_**

```python
# forward pass
W = np.random.randn(5, 10) # 5x10
X = np.random.randn(10, 3) # 10x3
D = W.dot(X) #  5x3

# backward pass
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T)
dX = W.T.dot(dD)

```

*calculus
*algebra
*geometry 
