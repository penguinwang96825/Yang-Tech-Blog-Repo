---
title: Principal Component Analysis Derivation
top: false
cover: false
toc: true
mathjax: true
date: 2021-03-13 22:07:26
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/13/2021-03-13-principal-component-analysis-derivation/wallhaven-rd3pjw.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/13/2021-03-13-principal-component-analysis-derivation/wallhaven-rd3pjw.jpg?raw=true
summary: Principal Component Analysis (PCA) is an important technique to understand in the fields of statistics and data science. It is a process of computing the principal components and utilising then to perform a change of basis on the data. For the purpose of visualisation, it is very hard to visulaise and understand the data in high dimensions, this is where PCA comes to the rescue.
tags:
	- Data Science
	- Machine Learning
categories: Data Science
---

Principal Component Analysis (PCA) is an important technique to understand in the fields of statistics and data science. It is a process of computing the principal components and utilising then to perform a change of basis on the data. For the purpose of visualisation, it is very hard to visulaise and understand the data in high dimensions, this is where PCA comes to the rescue.

# Introduction

Principal Component Analysis or (PCA) is a widely used technique for dimensionality reduction of the large data set. It needs the knowledge of some linear algebra, such as vector projection, eigenvalues and eigenvectors, Lagrange multipliers, derivatives of a matrix, and covariance matrix.

## Derivation

Let's go ahead and get into it. Let's say we have N different vectors {% mathjax %} x_{1} {% endmathjax %} to {% mathjax %} x_{N} {% endmathjax %} with dimension of D. Our goal of course in PCA is dimensionality reduction. So we want to map the space which has dimensionality D onto a space which has dimensionality M, where M has to be less than D by all means. That's the point of dimensionality reduction.

It looks like a scary and daunting task, let's take a whole step back and look at a simpler picture. Our objective is that we want to maximise the variance of the projections onto some dimensional space. In other words, we have this D dimensional space that contains all this information and all this data, we want to reduce its dimensionality, but we want to do it in a clever way, we want to do it onto a space so that we preserve as much of the information (original variation) while reducing the dimensions.

### Projection

The projection of {% mathjax %} x_{i} {% endmathjax %} onto a potential {% mathjax %} u {% endmathjax %} vector can be written as

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	proj_{u} x_i = \frac{u^T x_i}{\lVert u \rVert} u
	{% endmathjax %}
</div>

where u is a unit vector, so its length is 1, and we can get 

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	proj_{u} x_i = u^T x_i u
	{% endmathjax %}
</div>

Finally, we know that our mean of projections among all the data is {% mathjax %} u^T \overline{x} u {% endmathjax %}, since the mean is a linear operation, it behaves in the exact same way.

### Variance

Going back to our objective, our goal is to maximise the variance of the projected data. By the definition of the variance, 

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\displaystyle Var(X) = \frac{1}{N} \sum_{n=1}^{N} (u^T x_n - u^T \overline{x})^2
	{% endmathjax %}
</div>

Next, 

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\displaystyle Var(X) = \frac{1}{N} \sum_{n=1}^{N} (u^T (x_n - \overline{x}))^2
	{% endmathjax %}
</div>

And we expand it, 

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\displaystyle Var(X) = \frac{1}{N} \sum_{n=1}^{N} u^T (x_n - \overline{x}) (x_n - \overline{x})^T u
	{% endmathjax %}
</div>

Next, 

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\displaystyle Var(X) = \frac{1}{N} u^T \sum_{n=1}^{N} (x_n - \overline{x}) (x_n - \overline{x})^T u
	{% endmathjax %}
</div>

where {% mathjax %} \frac{1}{N} \sum_{n=1}^{N} (x_n - \overline{x}) (x_n - \overline{x})^T {% endmathjax %} is the closed form of the covariance matrix, so we then left

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	Var(X) = u^T S u
	{% endmathjax %}
</div>

### Lagrange Multipler

In this part, we want to maximise the variance of the projections which as we found is {% mathjax %} u^T S u {% endmathjax %} subject to the constraint {% mathjax %} u^T u = 1 {% endmathjax %}, and u means a unit vector. Using the power of Lagrange multipler, we have a new objective function, which looks like {% mathjax %} u^T S u + \lambda (1 - u^T u) {% endmathjax %}.

We are just going to take the derivative of this line with respect to vector {% mathjax %} u {% endmathjax %}, so we get 

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\frac{d}{d u} u^T S u + \lambda (1 - u^T u) = 2Su - \lambda \cdot 2u = 0
	{% endmathjax %}
</div>

We get 

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	S u = \lambda u
	{% endmathjax %}
</div>

This means that for u, whatever direction we choose to project on is going to have to be an eigenvector of the covariance matrix S, because this is exactly the definition of an eigenvector. But there's lots of eigenvectors and eigenvalues, what eigenvector and what eigenvalue should we use?

### Eigenvectors and Eigenvalues

To figure this out, we know that

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	u^T S u = u^T (\lambda u) = \lambda
	{% endmathjax %}
</div>

If we want the maximum value of {% mathjax %} u^T S u {% endmathjax %}, then we should select the dominant eigenvalue for the variance of the projected data.

To be more general, if we want to project the data onto more than just one dimension, we have to figure out what is the second biggest eigenvalue, and we use the second eigenvector corresponding to the second biggest eigenvalue, etc. You just go down in line for whatever many different components you want to end up in.

# Conclusion

In this article, we understand the moving parts behind Principal Component Analysis (PCA), I believe this will give you some insight into what's actually happening. I hope you are able to follow this article, stay tuned! Bye!