---
title: Softmax and Cross-Entropy
top: false
cover: false
toc: true
mathjax: true
date: 2021-04-18 16:08:44
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/04/18/2021-04-18-softmax-and-cross-entropy/wallhaven-n6xk36.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/04/18/2021-04-18-softmax-and-cross-entropy/wallhaven-n6xk36.jpg?raw=true
summary: I'm trying to implement neural network from scratch in Python recently. Considering to solve multi-class classification problem using neural network, I try to create a simple neural network. The most important thing in neural network is backpropagation. Backpropagation is an algorithm for supervised learning of artificial neural networks using gradient descent. I want to find the derivation of cross-entropy loss function with softmax activation function, so this article will record the formula I calculated. As for the rest, I will discuss it in the future.
tags:
	- Neural Network
	- Python
	- Calculus
categories: NLP
---

# Introduction

I'm trying to implement neural network from scratch in Python recently. Considering to solve multi-class classification problem using neural network, I try to create a simple neural network. The most important thing in neural network is backpropagation. Backpropagation is an algorithm for supervised learning of artificial neural networks using gradient descent. I want to find the derivation of cross-entropy loss function with softmax activation function, so this article will record the formula I calculated. As for the rest, I will discuss it in the future.

# Softmax Function

Softmax is a generalization of the logistic function to multiple dimensions. In general, the standard (unit) softmax function {% mathjax %} \sigma {% endmathjax %} : {% mathjax %} \mathbb{R} \rightarrow [0, 1]^{K} {% endmathjax %} is defined by the formula:

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\displaylines{\sigma(\mathbf{z})_{i} = \frac{e^{z_{i}}}{\Sigma_{j=1}^{K} e^{z_{j}}}}
	{% endmathjax %}
</div>

where {% mathjax %} \mathbf{z} = (z_i, \ldots, z_{K}) \in \mathbb{R}^{K} {% endmathjax %} for {% mathjax %} i = 1, \ldots, K {% endmathjax %}.

In simple words, it applies the standard exponential function to each element {% mathjax %} z_{i} {% endmathjax %} of the input vector {% mathjax %} \mathbf{z} {% endmathjax %} and normalises these values by dividing by the sum of all these exponentials; this normalization ensures that the sum of the components of the output vector {% mathjax %} \sigma(\mathbf{z}) {% endmathjax %} is 1.

In Python, we code the softmax function as follow:

```python
import numpy as np

def softmax(z):
    exps = np.exp(z)
    return exps / np.sum(exps)
```

The softmax function is actually numerically well-behaved. It has only positive terms, so we needn't worry about loss of significance, and the denominator is at least as large as the numerator, so the result is guaranteed to fall between 0 and 1. The only accident that might happen is over- or under-flow in the exponentials. Overflow of a single or underflow of all elements of {% mathjax %} \mathbf{z} {% endmathjax %} will render the output more or less useless. To make our softmax function numerically stable, we simply normalize the values in the vector, by multiplying the numerator and denominator with a constant {% mathjax %} \mathbf{C} {% endmathjax %}.

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\displaylines{\sigma(\mathbf{z})_{i} 
		&= \frac{e^{z_{i}}}{\Sigma_{j=1}^{K} e^{z_{j}}}\\
		&= \frac{C e^{z_{i}}}{C \Sigma_{j=1}^{K} e^{z_{j}}}\\
		&= \frac{e^{z_{i}+log(C)}}{\Sigma_{j=1}^{K} e^{z_{j}+log(C)}}\\
	}
	{% endmathjax %}
</div>

We can choose an arbitrary value for {% mathjax %} log(C) {% endmathjax %} term, but generally {% mathjax %} log(C) = -max(z) {% endmathjax %} is chosen. This will avoid overflowing and resulting in `nan`.

```python
def stable_softmax(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)
```

## Derivative of Softmax

One could also argue that in theory, using a deep network with a softmax function on top can represent any N-class probability function over the feature space. For this we need to calculate the derivative or gradient and pass it back to the previous layer during backpropagation.

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\frac{\partial}{\partial z_{j}} \frac{e^{z_{i}}}{\displaystyle\sum_{j=1}^{K} e^{z_{j}}}
	{% endmathjax %}
</div>

Let's take an easy example, we set {% mathjax %} \mathbf{z} = [z_1, z_2, z_3] {% endmathjax %}, so we know that {% mathjax %} softmax([z_1, z_2, z_3]) = y_1, y_2, y_3 {% endmathjax %}. And we can get {% mathjax %} y_1 = \frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}} {% endmathjax %}, {% mathjax %} y_2 = \frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_3}} {% endmathjax %}, {% mathjax %} y_3 = \frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}} {% endmathjax %}. First, we caclulate {% mathjax %} \frac{\partial y_1}{\partial z_1} {% endmathjax %}, {% mathjax %} \frac{\partial y_1}{\partial z_2} {% endmathjax %}, {% mathjax %} \frac{\partial y_1}{\partial z_3} {% endmathjax %}.

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\frac{\partial y_1}{\partial z_1} = \frac{e^{z_1} (e^{z_1}+e^{z_2}+e^{z_3}) - e^{z_1}e^{z_1}}{(e^{z_1}+e^{z_2}+e^{z_3})^{2}} = y_1 - \frac{e^{z_1}e^{z_1}}{(e^{z_1}+e^{z_2}+e^{z_3})^{2}} = y_1 - y_1^2
	{% endmathjax %}
</div>

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\frac{\partial y_1}{\partial z_2} = \frac{-e^{z_1}e^{z_2}}{(e^{z_1}+e^{z_2}+e^{z_3})^{2}} = - y_1 y_2
	{% endmathjax %}
</div>

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\frac{\partial y_1}{\partial z_3} = \frac{-e^{z_1}e^{z_2}}{(e^{z_1}+e^{z_3}+e^{z_3})^{2}} = - y_1 y_3
	{% endmathjax %}
</div>

Similarly, we can get {% mathjax %} \frac{\partial y_2}{\partial z_1} {% endmathjax %}, {% mathjax %} \frac{\partial y_2}{\partial z_2} {% endmathjax %}, {% mathjax %} \frac{\partial y_2}{\partial z_3} {% endmathjax %}, {% mathjax %} \frac{\partial y_3}{\partial z_1} {% endmathjax %}, {% mathjax %} \frac{\partial y_3}{\partial z_2} {% endmathjax %}, {% mathjax %} \frac{\partial y_3}{\partial z_3} {% endmathjax %}.

So, the derivative of the softmax function is given as {% mathjax %} \frac{\partial y_i}{\partial z_j} = y_{i} (\delta_{ij} - y_{j}) {% endmathjax %} where {% mathjax %} \delta_{ij} {% endmathjax %} is [Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta).

# Cross-Entropy

Cross-entropy is commonly used to quantify the difference between two probability distributions. Usually the ground truth distribution (the one that your machine learning algorithm is trying to match) is expressed in terms of a one-hot distribution. Cross-entropy loss is defined as 

<div style="display: flex;justify-content: center;">
	{% mathjax %}
		H(y, \hat y) = - \displaystyle\sum_{i} y_{i} log(\hat y_{i})
	{% endmathjax %}
</div>

For example, suppose for a specific training instance, the true label is B (out of the possible labels A, B, and C). The one-hot distribution for this training instance is therefore [0, 1, 0]. You can interpret the above true distribution to mean that the training instance has 0% probability of being class A, 100% probability of being class B, and 0% probability of being class C. Now, suppose your machine learning algorithm predicts the following probability distribution [0.228, 0.619, 0.153]. Use the formula we can get 

<div style="display: flex;justify-content: center;">
	{% mathjax %}
		H = - (0.0*log(0.228) + 1.0*log(0.619) + 0.0*log(0.153)) = 0.479
	{% endmathjax %}
</div>

```python
def cross_entropy(y, yhat):
    loss = 0
    for s1, s2 in zip(y, yhat):
        log_likelihood = - np.multiply(s1, np.log(s2))
        loss += np.sum(log_likelihood)
    loss = loss / y.shape[0]
    return loss
```

# Derivative of Cross-Entropy with Softmax

Now we use the derivative of softmax that we derived earlier to derive the derivative of the cross entropy loss function.

<div style="display: flex;justify-content: center;">
	{% mathjax %}
		L = - \displaystyle\sum_{i} y_{i} log(\hat y_{i})
	{% endmathjax %}
</div>

We want to compute the derivative of {% mathjax %} \mathbf{L} {% endmathjax %} with respect to {% mathjax %} z {% endmathjax %}.

<div style="display: flex;justify-content: center;">
	{% mathjax %}
		\frac{\partial L}{\partial z_i}
		 = - \displaystyle\sum_{k} y_{k} \frac{\partial log(\hat y_{k})}{\partial z_i}
		 = - \displaystyle\sum_{k} y_{k} \frac{\partial log(\hat y_{k})}{\partial \hat y_{k}} \frac{\partial \hat y_{k}}{\partial z_i}
		 = - \displaystyle\sum_{k} y_{k} \frac{1}{\hat y_{k}} \frac{\partial \hat y_{k}}{\partial z_i}
	{% endmathjax %}
</div>

From derivative of softmax we derived earlier,

<div style="display: flex;justify-content: center;">
	{% mathjax %}
		\frac{\partial L}{\partial z_i}
		 = - y_{i} (1 - \hat y_{i}) - \displaystyle\sum_{k \neq i} y_{k} \frac{1}{\hat y_{k}} (- \hat y_{k} \hat y_{i})
		 = - y_{i} (1 - \hat y_{i}) - \displaystyle\sum_{k \neq i} y_{k} \hat y_{i}
		 = \hat y_{i} (y_{i} + \displaystyle\sum_{k \neq i} y_{k}) - y_{i}
		 = \hat y_{i} - y_{i}
	{% endmathjax %}
</div>

which is a very simple and elegant expression.

# Conclusion

Real-world neural networks are capable of solving multi-class classification problems. In this article, we saw how I calculate the derivative of cross-entropy loss with softmax activation function. In the future, I will implement a simple feed-forward neural network using numpy only. With this computation beforehand, it will make the coding more easy to implement.

# References

1. https://stackoverflow.com/questions/42599498/numercially-stable-softmax
2. https://deepnotes.io/softmax-crossentropy
3. https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
4. http://cs231n.github.io/convolutional-networks/
5. https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/