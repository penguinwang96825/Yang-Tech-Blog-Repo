---
title: Eigenvectors from Eigenvalues
top: false
cover: false
toc: true
mathjax: true
date: 2021-03-15 14:11:42
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/15/2021-03-15-eigenvectors-from-eigenvalues/wallhaven-9mxz8k.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/15/2021-03-15-eigenvectors-from-eigenvalues/wallhaven-9mxz8k.jpg?raw=true
summary: This article is about implementing "Eigenvectors from eigenvalues" of Terence Tao's paper using Python and R. It's a amazing work and mathematics contribution from Terence Tao. It is an elegant non-evident result, which makes me so excited about it!
tags:
	- Python
	- R
	- Linear Algebra
categories: Mathematics
---

# Introduction

This article is about implementing "Eigenvectors from eigenvalues" of Terence Tao's paper using Python and R. It's a amazing work and mathematics contribution from Terence Tao. It is an elegant non-evident result, which makes me so excited about it!

# Eigenvector-eigenvalue Identity

According to Eigenvector-eigenvalue Identity Theorem, We have 

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\lVert v_{i, j} \rVert ^2 \displaystyle \prod_{k=1; k \neq i}^{n} (\lambda_{i}(A) - \lambda_{k}(A)) = \prod_{k=1}^{n-1} (\lambda_{i}(A) - \lambda_{k}(M_{j}))
	{% endmathjax %}
</div>

## Python Implementation

```python
def eigenvectors_from_eigenvalues(A, eig_val_A=None):
    """
    Implementation of Eigenvector-eigenvalue Identity Theorem
    Reference from https://dearxxj.github.io/post/7/

    Parameters:
        A: (n, n) Hermitian matrix (array-like)
        eig_val_A: (n, ) vector (float ndarray)
    Return: 
        eig_vec_A: Eigenvectors of matrix A
    """
    n = A.shape[0]
    # Produce eig_val_A by scipy.linalg.eigh() function
    if eig_val_A is None:
        eig_val_A, _ = eigh(A)
    eig_vec_A = np.zeros((n, n))
    start = time.time()
    for k in range(n):
        # Remove the k-th row
        M = np.delete(A, k, axis=0)
        # Remove the k-th column
        M = np.delete(M, k, axis=1)
        # Produce eig_val_M by scipy.linalg.eigh() function
        eig_val_M, _ = eigh(M)

        nominator = [np.prod(eig_val_A[i] - eig_val_M) for i in range(n)]
        denominator = [np.prod(np.delete(eig_val_A[i] - eig_val_A, i)) for i in range(n)]

        eig_vec_A[k, :] = np.array(nominator) / np.array(denominator)
    elapse_time = time.time() - start
    print("It takes {:.8f}s to compute eigenvectors using Eigenvector-eigenvalue Identity.".format(elapse_time))
    return eig_vec_A
```

Test `eigenvectors_from_eigenvalues()` on matrix A.

```python
A = np.array([[1, 1, -1], [1, 3, 1], [-1, 1, 3]])
eig_vec_A = eigenvectors_from_eigenvalues(A)
print(eig_vec_A)

start = time.time()
eig_val_A, eig_vec_A = eigh(A); eig_vec_A
print("\nIt takes {:.8f}s to compute eigenvectors using scipy.linalg.eigh() function.".format(time.time()-start))
print(eig_vec_A)
```

---

```console
It takes 0.00070190s to compute eigenvectors using Eigenvector-eigenvalue Identity.
[[0.66666667 0.33333333 0.        ]
 [0.16666667 0.33333333 0.5       ]
 [0.16666667 0.33333333 0.5       ]]

It takes 0.00016832s to compute eigenvectors using scipy.linalg.eigh() function.
[[ 0.81649658 -0.57735027  0.        ]
 [-0.40824829 -0.57735027  0.70710678]
 [ 0.40824829  0.57735027  0.70710678]]
```

## R Implementation

```R
A = matrix(c(1, 1, -1, 1, 3, 1, -1, 1, 3), 3, 3)

vec_from_val = function(A){
  # A should be Hermitian matrix
  n = sqrt(length(A))
  Aeig = eigen(A)$values
  # V is eigenvecters matrix
  V = matrix(ncol=n, nrow=n)
  
  for (i in 1:n){
    AM = Aeig[-i]
    for (j in 1:n){
      # Minors matrix B
      B = A[-j, ]
      B = B[, -j]
      Beig = eigen(B)$values

      down = 1; up = 1
      # n_0 is the dimension of B
      n_0 = n - 1
      
      for (k in 1:n_0){
        down = down * (Aeig[i] - AM[k])
        up = up * (Aeig[i] - Beig[k])
      }
      
      V[i, j] = up / down
    }
  }
  
  return(t(V))
}

vec_from_val(A)
```

# Conclusion

1. The eigenvector-eigenvalue identity only yields information about the magnitude of the components of a given eigenvector, but does not directly reveal the phase of these components. Otherwise, the eigenvector-eigenvalue identity may be more computationally feasible only if one has an application that requires only the component magnitudes.
2. It would be a computationally intensive task in general to compute all n-1 eigenvalues of each of the n minors matrices.
3. An additional method would then be needed to calculate the signs of these components of eigenvectors.
4. It has not been seen that the eigenvector-eigenvalue identity has better speed at computing eigenvectors compared to `scipy.linalg.eigh()` function.

## References

1. [[link](https://arxiv.org/pdf/1908.03795.pdf)] Terence Tao, Eigenvectors from eigenvalues: a survey of a basic identity in linear algebra.
2. [[link](https://www.ias.ac.in/article/fulltext/jcsc/101/06/0499-0517)] Asok K. Mukherjee and Kali Kinkar Datta. Two new graph-theoretical methods for generation of eigenvectors of chemical graphs
3. [[link](https://arxiv.org/pdf/1907.02534.pdf)] Peter B Denton, Stephen J Parke, and Xining Zhang. Eigenvalues: the Rosetta Stone for Neutrino Oscillations in Matter
4. https://math.stackexchange.com/questions/3436475/how-to-get-eigenvectors-from-eigenvalues
5. https://terrytao.wordpress.com/2019/08/13/eigenvectors-from-eigenvalues/