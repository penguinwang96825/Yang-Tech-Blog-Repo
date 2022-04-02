---
title: Calculate the Singular Value Decomposition
top: false
cover: false
toc: true
mathjax: true
date: 2021-03-16 20:37:22
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/16/2021-03-16-calculate-singular-value-decomposition/wallhaven-gjoplq.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/16/2021-03-16-calculate-singular-value-decomposition/wallhaven-gjoplq.jpg?raw=true
summary: Singular Value Decomposition (SVD) is a widely used technique to decompose a matrix into several component matrices, exposing many of the useful and interesting properties of the original matrix.
tags:
	- Linear Algebra
	- Data Science
categories: Data Science
---

# Introduction

Singular Value Decomposition (SVD) is a widely used technique to decompose a matrix into several component matrices, exposing many of the useful and interesting properties of the original matrix.

# Computation

The aim of this article is to find the singular value decompostiion of the given matrix. Take the below matrix for example.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
        C = 
        \left[ 
        	{\begin{array}{c}
        	5 & 5 \\
        	-1 & 7 \\
        	\end{array} } 
        \right]
    {% endmathjax %}
</div>

Let's point out here, it's actually very easy to find SVD for every matrices. So, what does the SVD look like? What do we want to end up with? The answer is we want a decomposition {% mathjax %} C = U \Sigma V^T {% endmathjax %}, where U and V are going to be orthogonal matrices, that is, their columns are orthonormal sets. Sigmas is going to be a diagnal matrix with non-negative entries.

In order to compute these matrices, we need two equations.

- {% mathjax %} C^T C = V \Sigma^T \Sigma V^T {% endmathjax %} (because {% mathjax %} (AB)^T=B^TA^T {% endmathjax %} and {% mathjax %} U {% endmathjax %} is an orthogonal matrix)
- {% mathjax %} CV = U \Sigma {% endmathjax %}

These are the two equations we need to use to find V, {% mathjax %} \Sigma {% endmathjax %}, and U.

Let's start with the first one, 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
        C^T C = 
        \left[ 
        	{\begin{array}{c}
        	5 & -1 \\
        	5 & 7 \\
        	\end{array} } 
        \right]
        \left[ 
        	{\begin{array}{c}
        	5 & 5 \\
        	-1 & 7 \\
        	\end{array} } 
        \right]
        = 
        \left[ 
        	{\begin{array}{c}
        	26 & 18 \\
        	18 & 74 \\
        	\end{array} } 
        \right]
    {% endmathjax %}
</div>

Now, what you notice about this equation is this is just a diagnalisation of {% mathjax %} C^T C {% endmathjax %}, so we need to find the eigenvalues, and those wil be the entries of {% mathjax %} \Sigma^T \Sigma {% endmathjax %}, and the eigenvectors will be the columns of V matrix.

Next, we look at the determinant of {% mathjax %} C^T C - \lambda I {% endmathjax %}, which will be

<div style="display: flex;justify-content: center;">
    {% mathjax %}
        det(C^T C - \lambda I) = det(
        	\left[ 
	        	{\begin{array}{c}
	        	26 - \lambda & 18 \\
	        	18 & 74 - \lambda \\
	        	\end{array} } 
	        \right]) = \lambda^2 - 100 \lambda + 160 = (\lambda - 20)(\lambda - 80)
    {% endmathjax %}
</div>

So the eigenvalues are 20 and 80.

In order to calculate the corresponding eigenvectors, we first take 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
        C^T C - 20 I = 
        	\left[ 
	        	{\begin{array}{c}
	        	6 & 18 \\
	        	18 & 54 \\
	        	\end{array} } 
	        \right]
    {% endmathjax %}
</div>

Next, we need to find the [null space](https://www.geeksforgeeks.org/null-space-and-nullity-of-a-matrix/) of this matrix. Therefore, we get the eigenvector for  {% mathjax %} \lambda = 20 {% endmathjax %} is {% mathjax %} v_1 = (-3, 1)^T {% endmathjax %}. We want it to be a unit vector, remember the columns of V should be unit vectors because they're orthonormal. Therefore, {% mathjax %} v_1 = (\frac{-3}{\sqrt{10}}, \frac{1}{\sqrt{10}})^T {% endmathjax %}. Similarly, we get {% mathjax %} v_2 = (\frac{1}{\sqrt{10}}, \frac{3}{\sqrt{10}})^T {% endmathjax %} for {% mathjax %} \lambda = 80 {% endmathjax %}.

Add these vectors and make them the columns of V matrix, then we can get 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
        V = 
        	\left[ 
	        	{\begin{array}{c}
	        	\frac{-3}{\sqrt{10}} & \frac{1}{\sqrt{10}} \\
	        	\frac{1}{\sqrt{10}} & \frac{3}{\sqrt{10}} \\
	        	\end{array} } 
	        \right], 
	    \Sigma = 
	    	\left[ 
	        	{\begin{array}{c}
	        	2 \sqrt{5} & 0 \\
	        	0 & 4 \sqrt{5} \\
	        	\end{array} } 
	        \right]
    {% endmathjax %}
</div>

Good, now we can get these two of the three parts of SVD. The last thing we need to find is U matrix. For that, we need to use the second equation {% mathjax %} CV = U \Sigma {% endmathjax %}. So, 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
        \left[ 
        	{\begin{array}{c}
        	5 & 5 \\
        	-1 & 7 \\
        	\end{array} } 
        \right] 
        \left[ 
        	{\begin{array}{c}
        	\frac{-3}{\sqrt{10}} & \frac{1}{\sqrt{10}} \\
	        \frac{1}{\sqrt{10}} & \frac{3}{\sqrt{10}} \\
        	\end{array} } 
        \right] = 
        \left[ 
        	{\begin{array}{c}
        	-\sqrt{10} & 2 \sqrt{10} \\
	        \sqrt{10} & 2 \sqrt{10} \\
        	\end{array} } 
        \right] = 
        \left[ 
        	{\begin{array}{c}
        	-\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
	        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        	\end{array} } 
        \right]
        \left[ 
        	{\begin{array}{c}
        	2\sqrt{5} & 0 \\
	        0 & 4 \sqrt{10} \\
        	\end{array} } 
        \right] 
    {% endmathjax %}
</div>

So now, here's our U matrix.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
        U = 
        	\left[ 
	        	{\begin{array}{c}
	        	-\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
		        \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
	        	\end{array} } 
	        \right]
    {% endmathjax %}
</div>

Finally we have all three matrices U, V, and {% mathjax %} \Sigma {% endmathjax %}.

# Conclusion

This is a good illustration of how to find SVD by hand. Please stay tuned as this blog will be updated regularly!