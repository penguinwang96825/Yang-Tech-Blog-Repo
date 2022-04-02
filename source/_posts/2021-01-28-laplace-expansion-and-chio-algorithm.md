---
title: Laplace Expansion and Chiò Condensation
date: 2021-01-28 09:25:00
author: Yang Wang
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/01/28/2021-01-28-laplace-expansion-and-chio-algorithm/call-me-lambh.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/01/28/2021-01-28-laplace-expansion-and-chio-algorithm/course-image.png?raw=true
summary: Determinants are mathematical objects which have applications in engineering mathematics. For example, they can be used in the solution of simultaneous equations, and to evaluate vector products. Determinants can also be used to see if a system of $n$ linear equations in $n$ variables has a unique solution. There are several ways to calculate determinant, however, today I'm going to introduce another way of computing determinants Chio Identity.
categories: Mathematics
tags:
  - Python
  - Linear Algebra
---

Determinants are mathematical objects which have applications in engineering mathematics. For example, they can be used in the solution of simultaneous equations, and to evaluate vector products. Determinants can also be used to see if a system of *n* linear equations in *n* variables has a unique solution. There are several ways to calculate determinant, however, today I'm going to introduce another way of computing determinants: Chiò Identity.

# Introduction

In 1853 Felice (Félix) Chiò (1813–1871) published his short "Mémoire sur les fonctions connues sous le noms De Résultantes Ou De Déterminans". In this article, I first give a way of evaluating determinants by Laplace Expansion, and explicitly comparing Chiò Identity to this.

# Laplace Expansion Method

In linear algebra, the Laplace expansion, named after Pierre-Simon Laplace, also called cofactor expansion, is an expression for the determinant of an {% mathjax %} n \times n {% endmathjax %} matrix {% mathjax %} A {% endmathjax %} that is a weighted sum of the determinants of {% mathjax %} n {% endmathjax %} sub-matrices (or minors) of {% mathjax %} M {% endmathjax %}, each of size {% mathjax %} (n − 1) \times (n − 1) {% endmathjax %}. The {% mathjax %} i {% endmathjax %}, {% mathjax %} j {% endmathjax %} cofactor of the matrix {% mathjax %} A {% endmathjax %} is the scalar {% mathjax %} C_{ij} {% endmathjax %} defined by {% mathjax %} C_{ij}=(-1)^{i+j}M_{ij} {% endmathjax %}, where {% mathjax %} M_{ij} {% endmathjax %} is the {% mathjax %} i {% endmathjax %}, {% mathjax %} j {% endmathjax %} minor of {% mathjax %} A {% endmathjax %}, that is, the determinant of the {% mathjax %} (n − 1) \times (n − 1) {% endmathjax %} matrix that results from deleting the {% mathjax %} i {% endmathjax %}-th row and the {% mathjax %} j {% endmathjax %}-th column of {% mathjax %} A {% endmathjax %}.

```python
class LaplaceDeterminants:
    def __init__(self):
        pass
    
    def minor_matrix(self, A, i, j):
        # Delete i-th row
        sub_A = np.delete(A, i, 0)
        # Delete j-th column
        sub_A = np.delete(sub_A, j, 1)
        return sub_A
    
    def calculate(self, A):
        n, m = A.shape
        if not n == m: 
            raise Exception("Must be a square matrix!")
        if n == 2:
            return A[0][0]*A[1][1] - A[1][0]*A[0][1]
        det = 0
        for i in range(n):
            M = self.minor_matrix(A, 0, i)
            det += (-1)**i * A[0][i] * self.calculate(M)
        return det
```

# Chiò Condensation Method

The statement of Chiò Condensation is: let {% mathjax %} A=(a_{ij}) {% endmathjax %} be an {% mathjax %} n \times n {% endmathjax %} matrix, and let {% mathjax %} a_{11} \neq 0 {% endmathjax %}. Replace each element {% mathjax %} a_{ij} {% endmathjax %} in the {% mathjax %} (n-1) \times (n-1) {% endmathjax %} sub-matrix, let's called it {% mathjax %} D {% endmathjax %}, of {% mathjax %} A {% endmathjax %} obtained by deleting the {% mathjax %} i {% endmathjax %}th row and {% mathjax %} j {% endmathjax %}th column of {% mathjax %} A {% endmathjax %} by:

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    D = 
    \left[ {\begin{array}{cc}
    a_{ij} & a_{in} \\
    a_{nj} & a_{nn} \\
    \end{array} } \right]
    {% endmathjax %}
</div>

Then {% mathjax %} det(A)= \frac{1}{a_{nn}^{n-2}} \cdot det(D) {% endmathjax %}.

```python
class ChioDeterminants:
    def __init__(self):
        pass
    
    def calculate(self, A):
        n, m = A.shape
        if not n == m: 
            raise Exception("Must be a square matrix!")
        if n == 2:
            return A[0][0]*A[1][1] - A[1][0]*A[0][1]
        if A[-1][-1] == 0:
            for i in range(n):
                if A[0][i] != 0:
                    A[:, [i, n-1]] = A[:, [n-1, i]]
                    A[[0, n-1], :] = A[[n-1, 0], :]
                    break
                else:
                    return 0
        D = np.zeros(shape=(n-1, n-1))
        for i in range(n-1):
            for j in range(n-1):
                D[i][j] = A[i][j]*A[-1, -1] - A[-1][j]*A[i][-1]
        det = (1/A[-1][-1]**(n-2)) * self.calculate(D)
        return det
```

# Performance

```python
def test_laplace(n_samples=50000):
    algo = LaplaceDeterminants()
    for i in range(n_samples):
        A = np.random.rand(5, 5)
        det = algo.calculate(A)

def test_chio(n_samples=50000):
    algo = ChioDeterminants()
    for i in range(n_samples):
        A = np.random.rand(5, 5)
        det = algo.calculate(A)
```

{% asset_img perf1.png %}

What if we also compare Chiò Condensation Method to `numpy` and `scipy`? They both computed determinants via LU factorization, relying on BLAS and [LAPACK](http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html) to provide efficient low level implementations of standard linear algebra algorithms.

```python
def test_numpy(n_samples=50000):
    for i in range(n_samples):
        A = np.random.rand(5, 5)
        det = np.linalg.det(A)

def test_scipy(n_samples=50000):
    for i in range(n_samples):
        A = np.random.rand(5, 5)
        det = linalg.det(A)
```

{% asset_img perf2.png %}

Quantstart has implemented an LU Decomposition directly over [here](https://www.quantstart.com/articles/LU-Decomposition-in-Python-and-NumPy/), which does not rely on any external libraries.

# Conclusion

Clearly, Chiò Condensation Method is much quicker than Laplace Expansion Method by minors, which yeilds complexity computation of {% mathjax %} O(n!) {% endmathjax %}. As an alternative method for hand-calculating determinants, therefore, Chiò's method is quite effective. For numerical computations of large determinants on a computer, however, Chiò's method is not so efficient as other methods such as, for example, Gaussian elimination, because of certain difficulties with round-off errors. In addition, Chiò's method requires approximately {% mathjax %} \frac{2}{3}n^3 {% endmathjax %} multiplications, whereas Gaussian elimination requires approximately {% mathjax %} \frac{1}{3}n^3 {% endmathjax %}. 

## References

1. https://www.sciencedirect.com/science/article/pii/S0024379514002249
2. https://www.codeformech.com/determinant-linear-algebra-using-python/
3. https://en.wikipedia.org/wiki/Laplace_expansion
4. https://stackoverflow.com/questions/16636858/complexity-computation-of-a-determinant-recursive-algorithm
5. https://www.youtube.com/watch?v=UlWcofkUDDU.
6. Fuller, L. E., & Logan, J. D. On the Evaluation of Determinants by Chiò Method, 1975
