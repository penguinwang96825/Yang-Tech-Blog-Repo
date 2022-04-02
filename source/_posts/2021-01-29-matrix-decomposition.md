---
title: Demystify Matrix Decomposition
date: 2021-01-29 09:25:00
author: Yang Wang
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/01/29/2021-01-29-matrix-decomposition/kelly-sikkema.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/01/29/2021-01-29-matrix-decomposition/what-is-decomposition.png?raw=true
summary: The most important application for decomposition is in data fitting. The following discussion is mostly presented in terms of different methods of decomposition for linear function.
categories: Mathematics
tags:
  - Python
  - MATLAB
  - Linear Algebra
---

The most important application for decomposition is in data fitting. The following discussion is mostly presented in terms of different methods of decomposition for linear function.

# Introduction

The method of least squares is a standard approach in regression analysis to approximate the solution of overdetermined systems (sets of equations in which there are more equations than unknowns) by minimizing the sum of the squares of the residuals made in the results of every single equation. An matrix decomposition is a way of reducing a matrix into its constituent parts. It's an approach that can specify more complex matrix operation that can be performed on the decomposed matrx rather than on the origin matrix itself. There are various matrix decomposition methods, such as LU decomposition, QR decomposition, SVD decomposition, and Cholesky decomposition, etc.

# LU Decomposition

> Least Square: Let {% mathjax %} X \in \mathbb{R}^{m \times n} {% endmathjax %} with {% mathjax %} m>n {% endmathjax %}, {% mathjax %} y \in \mathbb{R}^{m \times 1} {% endmathjax %}, and {% mathjax %} \beta \in \mathbb{R}^{n \times 1} {% endmathjax %}. We aim to solve {% mathjax %} y=X \beta {% endmathjax %} where {% mathjax %} \hat{\beta} {% endmathjax %} is the least square estimator. The least squares solution for {% mathjax %} \hat{\beta}=(X^{T} X)^{-1}X^{T}y {% endmathjax %} can obtained using different decomposition methods on {% mathjax %} X^{T} X {% endmathjax %}.

When using LU, we have {% mathjax %} \hat{\beta}=(X^{T} X)^{-1}X^{T}y=(LU)^{-1}X^{T}y {% endmathjax %}, decomposing the square matrix {% mathjax %} X^{T} X {% endmathjax %} into {% mathjax %} L {% endmathjax %} and {% mathjax %} U {% endmathjax %} components. The factors {% mathjax %} L {% endmathjax %} and {% mathjax %} U {% endmathjax %} are triangular matrices. A variation of this decomposition that is numerically more stable to solve in practice is called the PLU decomposition, or the LU decomposition with partial pivoting, where {% mathjax %} P {% endmathjax %} is a so-called permutation matrix, {% mathjax %} L {% endmathjax %} is lower triangular, and {% mathjax %} U {% endmathjax %} is upper triangular. Lower and upper triangular matrices are computationally easier than your typical invertible matrix. The matrix P is easy to deal with as well since it is mostly full of zeros.  This [video](https://www.youtube.com/watch?v=UlWcofkUDDU&ab_channel=Mathispower4u) explains how to find the LU decomposition of a square matrix using a shortcut involving the opposite of multipliers used when performing row operations. There is also another [posting](https://math.unm.edu/~loring/links/linear_s08/LU.pdf) describe LU and PLU factorisation with some examples.

## Pseudocode

**Step 1.** Start with three candidate matrices:

* {% mathjax %} U = M {% endmathjax %}
* {% mathjax %} L = 0_{n, n} {% endmathjax %}
* {% mathjax %} P^{T} = I_{n} {% endmathjax %}

where {% mathjax %} L {% endmathjax %} is a {% mathjax %} n \times n {% endmathjax %} zeros matrix and {% mathjax %} P_{T} {% endmathjax %} is a {% mathjax %} n \times n {% endmathjax %} identity matrix.

**Step 2.** For {% mathjax %} i=1, 2, \ldots, n-1 {% endmathjax %}, find the row {% mathjax %} j {% endmathjax %} with the largest entry in absolute value on or below the diagonal of the {% mathjax %} i {% endmathjax %}-row and swap rows {% mathjax %} i {% endmathjax %} and {% mathjax %} j {% endmathjax %} in all three matrices, {% mathjax %} P_{T} {% endmathjax %}, {% mathjax %} L {% endmathjax %}, and {% mathjax %} U {% endmathjax %}. If this maximum entry is zero, then terminate this loop and indicate that the matrix is singular (invertible).

**Step 3.** Inside the first loop, create a second for loop, for {% mathjax %} j=i+1, \ldots, n {% endmathjax %}, calculate the scalar value {% mathjax %} s= \frac{-u_{j, i}}{u_{i, i}} {% endmathjax %}. Next, add {% mathjax %} s {% endmathjax %} times row {% mathjax %} i {% endmathjax %} onto row {% mathjax %} j {% endmathjax %} in the matrix $U$ and set the entry {% mathjax %} L_{j, y}=-s {% endmathjax %}.

**Step 4.** Having iterated from {% mathjax %} i=1, 2, \ldots, n-1 {% endmathjax %}, finish by adding the identity matrix onto {% mathjax %} L=L+I_{n} {% endmathjax %}. These are the {% mathjax %} P_{T} {% endmathjax %}, {% mathjax %} L {% endmathjax %}, and {% mathjax %} U {% endmathjax %} matrices of the PLU decomposition of matrix {% mathjax %} M {% endmathjax %}.

```python
class Decomposition:
    """
    References
    ----------
    [1] https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/04LinearAlgebra/lup/
    [2] https://math.unm.edu/~loring/links/linear_s08/LU.pdf
    [3] https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html
    """
    def __init__(self):
        pass
    
    def plu(self, A):
        # Step 1. Inittiate three cadidate matrices
        n = A.shape[0]
        P = np.identity(n)
        L = np.identity(n)
        U = A.copy()
        
        PF = np.identity(n)
        LF = np.zeros(shape=(n, n))
        
        # Step 2. Loop over rows find the row with the largest entry in absolute
        # value on or below the diagonal of the i-th row
        for i in range(n-1):
            index = np.argmax(abs(U[i:, i]))
            index = index + i
            if index != i:
                P = np.identity(n)
                P[[index, i], i:n] = P[[i, index], i:n]
                U[[index, i], i:n] = U[[i, index], i:n] 
                PF = np.dot(P, PF)
                LF = np.dot(P, LF)
            L = np.identity(n)
            # Step 3. Calculate the scalar value
            for j in range(i+1, n):
                L[j, i]  = -(U[j, i] / U[i, i])
                LF[j, i] =  (U[j, i] / U[i, i])
            U = np.dot(L, U)
        # Step 4. Add identity matrix onto L
        np.fill_diagonal(LF, 1)
        return PF, LF, U
```

# QR decomposition

In linear algebra, a QR decomposition, also known as a QR factorization or QU factorization is a decomposition of a matrix {% mathjax %} A {% endmathjax %}, either square or rectangular, into a product {% mathjax %} A = QR {% endmathjax %} of an orthogonal matrix {% mathjax %} Q {% endmathjax %} and an upper triangular matrix {% mathjax %} R {% endmathjax %}. There are in fact a couple of methods to compute a QR decomposition. These include the the `Gram–Schmidt process`, `Householder transformations`, and `Givens rotations`.

If {% mathjax %} A \in \mathbb{R^{m \times n}} {% endmathjax %} has linearly independent columns, then it can factored as: 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
        A= 
        \left[ {\begin{array}{c}
        q_{1} & q_{2} & \ldots & q_{n} \\
        \end{array} } \right]
        \left[ {\begin{array}{ccc}
        R_{11} & R_{12} & \ldots & R_{1n}\\
        0 & R_{22} & \ldots & R_{2n} \\
        \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & \vdots & R_{nn} \\
        \end{array} } \right]
    {% endmathjax %}
</div>

where {% mathjax %} q_{1}, q_{2}, \ldots, q_{n} {% endmathjax %} are orthogonal {% mathjax %} m {% endmathjax %}-vectors ({% mathjax %} Q^{T}Q=I {% endmathjax %}), that is, {% mathjax %} \lVert q_{i} \rVert  = 1 {% endmathjax %}, {% mathjax %} q_{i}^{T}q_{j} = 0 {% endmathjax %} if {% mathjax %} i \neq j {% endmathjax %}. Moreover, diagonal elements {% mathjax %} R_{ii} {% endmathjax %} are nonzero ({% mathjax %} R {% endmathjax %} is nonsingular), and most definitions require {% mathjax %} R_{ii} > 0 {% endmathjax %}, this makes {% mathjax %} Q {% endmathjax %} and {% mathjax %} R {% endmathjax %} unique.

Before dive into how to calculate QR factorisation, we should know what problem or application we can tackle with or apply for.

* Linear equations
* Generalised linear regression model 
* Singular-value decomposition in the Jacobi-Kogbetliantz approach
* Automatic removal of an object from an image

## Algorithms for QR

1. Gram-Schmidt process
* Complexity is {% mathjax %} 2mn^{2} {% endmathjax %} flops
* Not recommended in practice (sensitive to rounding errors)
2. Modified Gram-Schmidt process
* Complexity is {% mathjax %} 2mn^{2} {% endmathjax %} flops
* Better numerical properties
3. Householder transformations
* Complexity is {% mathjax %} 2mn^{2}-\frac{2}{3}n^{3} {% endmathjax %} flops
* Represents {% mathjax %} Q {% endmathjax %} as a product of elementary orthogonal matrices
* The most widely used algorithm

### Gram-Schmidt Process

The goal of Gram-Schmidt process is to calculate orthogonal basis {% mathjax %} \vec{u_{1}}, \vec{u_{2}}, \cdots, \vec{u_{n}} {% endmathjax %} from original basis {% mathjax %} \vec{v_{1}}, \vec{v_{2}}, \cdots, \vec{v_{n}} {% endmathjax %}, and this can also be represented in summation notation:

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \vec{u_{k}} = \vec{v_{k}} - \sum_{i=1}^{k-1} \frac{\vec{v_{k}} \cdot \vec{u_{i}}}{\lVert \vec{u_{i}} \rVert^2} \vec{u_{i}}
    {% endmathjax %}
</div>


A full calculation process can be found in this youtube [video](https://www.youtube.com/watch?v=zHbfZWZJTGc&ab_channel=ProfessorDaveExplains) presented by Dave.

#### Matlab code

```matlab
function [Q, R] = gram_schmidt_qr(A)
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n, n);
    for j = 1:n
        R(1:j-1, j) = Q(:, 1:j-1)' * A(:, j);
        v = A(:, j) - Q(:, 1:j-1) * R(1:j-1, j);
        R(j, j) = norm(v);
        Q(:, j) = v / R(j, j);
    end;
```

#### Python code

```python
class Decomposition:
    def __init__(self):
        pass

    def gram_schmidt_qr(self, A):
        m, n = A.shape
        Q = np.zeros(shape=(m, n))
        R = np.zeros(shape=(n, n))
        for j in range(n):
            v = A[:, j]
            for i in range(j):
                q = Q[:, i]
                R[i, j] = q.dot(v)
                v = v - R[i, j] * q
            Q[:, j] = v / np.linalg.norm(v)
            R[j, j] = np.linalg.norm(v)
        return Q, R
```

### Modified Gram-Schmidt Process

In 1966 John Rice showed by experiments that the two different versions of the Gram–Schmidt orthogonalization, classical (CGS) and modified (MGS) have very different properties when executed in finite precision arithmetic. Only for n = 2 are CGS and MGS numerically equivalent.

Instead of computing the vector {% mathjax %} \vec{u_{k}} {% endmathjax %} as {% mathjax %} \vec{u_{k}} = \vec{v_{k}} - \sum_{i=1}^{k-1} \frac{\vec{v_{k}} \cdot \vec{u_{i}}}{\lVert \vec{u_{i}} \rVert^2} \vec{u_{i}} {% endmathjax %}, it is computed as:

<div style="display: flex;justify-content: center;">
    {% mathjax %}
        \begin{align}
            \displaylines{\vec{u_{k}}^{(1)} &= \vec{v_{k}} - proj_{\vec{u_{1}}} \vec{v_{k}} \\
                          \vec{u_{k}}^{(2)} &= \vec{u_{k}}^{(1)} - proj_{\vec{u_{2}}} \vec{u_{k}}^{(1)} \\
                          \vdots \\
                          \vec{u_{k}}^{(k-1)} &= \vec{u_{k}}^{(k-2)} - proj_{\vec{u_{k-1}}} \vec{u_{k}}^{(k-2)} \\
                          \vec{u_{k}} &= \frac{\vec{u_{k}}^{(k-1)}}{\lVert \vec{u_{k}}^{(k-1)} \rVert} }
        \end{align}
    {% endmathjax %}
</div>

where {% mathjax %} proj_{\vec{u}}\vec{v} = \frac{\vec{v} \cdot \vec{u}}{\lVert \vec{u} \rVert^2} \vec{u} {% endmathjax %}.

#### Matlab code

```matlab
function [Q, R] = modified_gram_schmidt_qr(A)
    [m, n] = size(A);
    Q = A;
    R = zeros(n);
    for k = 1:n
        R(k, k) = norm(Q(:, k));
        Q(:, k) = Q(:, k) / R(k, k);
        R(k, k+1:n) = Q(:,k)' * Q(:, k+1:n);
        Q(:, k+1:n) = Q(:, k+1:n) - Q(:, k) * R(k, k+1:n);
    end
```

#### Python code

```python
class Decomposition:
    def __init__(self):
        pass

    def modified_gram_schmidt_qr(self, A):
        m, n = A.shape
        Q = np.zeros(shape=(m, n))
        R = np.zeros(shape=(n, n))
        for j in range(0, n):
            R[j, j] = np.sqrt(np.dot(A[:, j], A[:, j]))
            Q[:, j] = A[:, j] / R[j, j]
            for i in range(j+1, n):
                R[j, i] = np.dot(Q[:, j], A[:, i])
                A[:, i] = A[:, i] - R[j, i] * Q[:, j]
        return Q, R
```

### Householder Transformations

Householder transformations are simple orthogonal transformations corresponding to reflection through a plane. Reflection across the plane orthogonal to a unit normal vector {% mathjax %} v {% endmathjax %} can be expressed in matrix form as

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    H = I - 2vv^T
    {% endmathjax %}
</div>

In particular, if we take {% mathjax %} u=x-s \lVert x \rVert e_{1} {% endmathjax %} where {% mathjax %} s= \pm 1 {% endmathjax %} and {% mathjax %} v=u/ \lVert u \rVert {% endmathjax %} then

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    Hx = (I - 2 \frac{uu^T}{u^Tu})x = s \lVert x \rVert e_{1}
    {% endmathjax %}
</div>

Let us first verify that this works:

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    u^Tx = (x-s \lVert x \rVert e_{1})^T x = \lVert x \rVert^2 - s x_{1} \lVert x \rVert
    {% endmathjax %}
</div>

and 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    u^Tu = (x-s \lVert x \rVert e_{1})^T(x-s \lVert x \rVert e_{1}) = \lVert x \rVert^2 - 2 s x_{1} \lVert x \rVert + \lVert x \rVert^2 \lVert e_1 \rVert^2 = 2 (\lVert x \rVert^2 - s x_1 \lVert x \rVert)
    {% endmathjax %}
</div>

so 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    u^Tu = 2 u^T x
    {% endmathjax %}
</div>

finally

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    Hx = x - 2 u \frac{u^T x}{u^T u} = x - u = s \lVert x \rVert e_1
    {% endmathjax %}
</div>


As a byproduct of this calculation, note that we have

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    u^Tu = -2 s \lVert x \rVert u_1
    {% endmathjax %}
</div>

where {% mathjax %} u_1 = x_1 - s \lVert x \rVert {% endmathjax %}; and if we define {% mathjax %} w = u/u_1 {% endmathjax %}, we have

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    H = I - 2 \frac{ww^T}{w^Tw} = I + \frac{su_1}{\lVert x \rVert} ww^T = I - \tau ww^T
    {% endmathjax %}
</div>

where {% mathjax %} \tau = -s u_1 / \lVert x \rVert {% endmathjax %}.

#### Matlab code

```matlab
function [Q,R] = householder_qr(A)
    [m, n] = size(A);
    Q = eye(m);
    R = A;
    I = eye(n);

    for j = 1:n-1
        x = R(j:n, j);
        v = -sign(x(1)) * norm(x) * eye(n-j+1, 1) - x;
        if norm(v) > 0
            v = v / norm(v);
            P = I;
            P(j:n, j:n) = P(j:n, j:n) - 2*v*v';
            R = P * R;
            Q = Q * P;
        end
    end
```

#### Python code

```python
class Decomposition:
    """
    https://stackoverflow.com/a/53493770/15048366
    """
    def __init__(self):
        pass

    def householder_vectorised(self, arr):
        v = arr / (arr[0] + np.copysign(np.linalg.norm(arr), arr[0]))
        v[0] = 1
        tau = 2 / (v.T @ v)
        return v, tau

    def householder_qr(self, A):
        m, n = A.shape
        Q = np.identity(m)
        R = A.copy()

        for j in range(0, n):
            v, tau = self.householder_vectorised(R[j:, j, np.newaxis])
            H = np.identity(m)
            H[j:, j:] -= tau * (v @ v.T)
            R = H @ R
            Q = H @ Q

        Q, R = Q[:n].T, np.triu(R[:n])
        for i in range(n):
            if R[i, i] < 0:
                Q[:, i] *= -1
                R[i, :] *= -1
                
        return Q, R
```

if {% mathjax %} m > n {% endmathjax %}:

* Full QR factorisation
{% asset_img full_qr.png %}

* Reduced QR factorisation
{% asset_img reduced_qr.png %}

# Linear Function

After decomposing matrix {% mathjax %} A {% endmathjax %}, you can write a function in python to solve a system

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    Ax = b
    {% endmathjax %}
</div>

using LU decomposition and QR decomposition. Your function should take {% mathjax %} A {% endmathjax %} and {% mathjax %} b {% endmathjax %} as input and return {% mathjax %} x {% endmathjax %}.

Function should include the following:

* Check that {% mathjax %} A {% endmathjax %} is not a singular matrix, that is, {% mathjax %} A {% endmathjax %} is invertible.
* Invert {% mathjax %} A {% endmathjax %} using different decomposition methods and solve
* Return {% mathjax %} x {% endmathjax %}.

```python
class Decomposition:
    
    def plu(self, A):
        n = A.shape[0]
        P = np.identity(n)
        L = np.identity(n)
        U = A.copy()
        
        PF = np.identity(n)
        LF = np.zeros(shape=(n, n))
        
        # Loop over rows
        for i in range(n-1):
            index = np.argmax(abs(U[i:, i]))
            index = index + i
            if index != i:
                P = np.identity(n)
                P[[index, i], i:n] = P[[i, index], i:n]
                U[[index, i], i:n] = U[[i, index], i:n] 
                PF = np.dot(P, PF)
                LF = np.dot(P, LF)
            L = np.identity(n)
            for j in range(i+1, n):
                L[j, i]  = -(U[j, i] / U[i, i])
                LF[j, i] =  (U[j, i] / U[i, i])
            U = np.dot(L, U)
        np.fill_diagonal(LF, 1)
        return PF, LF, U
    
    def gram_schmidt_qr(self, A):
        m, n = A.shape
        Q = np.zeros(shape=(m, n), dtype='float64')
        R = np.zeros(shape=(n, n), dtype='float64')
        for j in range(n):
            v = A[:, j]
            for i in range(j):
                q = Q[:, i]
                R[i, j] = q.dot(v)
                v = v - R[i, j] * q
            Q[:, j] = v / np.linalg.norm(v)
            R[j, j] = np.linalg.norm(v)
        return Q, R
    
    def modified_gram_schmidt_qr(self, A):
        n = A.shape[1]
        Q = np.array(A, dtype='float64')
        R = np.zeros((n, n), dtype='float64')
        for k in range(n):
            a_k = Q[:, k]
            R[k,k] = np.linalg.norm(a_k)
            a_k /= R[k, k]
            for i in range(k+1, n):
                a_i = Q[:, i]
                R[k,i] = np.transpose(a_k) @ a_i
                a_i -= R[k, i] * a_k
        return Q, R
    
    def householder_vectorised(self, arr):
        v = arr / (arr[0] + np.copysign(np.linalg.norm(arr), arr[0]))
        v[0] = 1
        tau = 2 / (v.T @ v)
        return v, tau

    def householder_qr(self, A):
        m, n = A.shape
        Q = np.identity(m)
        R = A.copy()

        for j in range(0, n):
            v, tau = self.householder_vectorised(R[j:, j, np.newaxis])
            H = np.identity(m)
            H[j:, j:] -= tau * (v @ v.T)
            R = H @ R
            Q = H @ Q

        Q, R = Q[:n].T, np.triu(R[:n])
        for i in range(n):
            if R[i, i] < 0:
                Q[:, i] *= -1
                R[i, :] *= -1
                
        return Q, R

def linear_function_solver(A, b, method="LU"):
    det = ChioDeterminants().calculate(A)
    factoriser = Decomposition()
    if det == 0:
        print("Matrix is singular!")
        return
    if method == "LU":
        P, L, U = factoriser.plu(A)
        z_1 = P.T @ b
        z_2 = np.linalg.inv(L) @ z_1
        x = np.linalg.inv(U) @ z_2
        return x
    elif method == "CGS":
        Q, R = factoriser.gram_schmidt_qr(A)
        x = np.linalg.inv(R) @ Q.T @ b
        return x
    elif method == "MGS":
        Q, R = factoriser.modified_gram_schmidt_qr(A)
        x = np.linalg.inv(R) @ Q.T @ b
        return x
    elif method == "HHT":
        Q, R = factoriser.householder_qr(A)
        x = np.linalg.inv(R) @ Q.T @ b
        return x
```

Let's check on four different approachs.

```python
A = np.array([
    [8, 6, 4, 1], 
    [1, 4, 5, 1], 
    [7, 4, 2, 5], 
    [1, 4, 2, 6]])
b = np.array([20, 12, 23, 19])

print("NP:  ", np.linalg.solve(A, b))
print("LU:  ", linear_function_solver(A, b, method="LU"))
print("CGS: ", linear_function_solver(A, b, method="CGS"))
print("MGS: ", linear_function_solver(A, b, method="MGS"))
print("HHT: ", linear_function_solver(A, b, method="HHT"))
```

---

```python
NP:   [1. 1. 1. 2.]
LU:   [1. 1. 1. 2.]
CGS:  [1. 1. 1. 2.]
MGS:  [1. 1. 1. 2.]
HHT:  [1. 1. 1. 2.]
```

# Conclusion

In this article, I implement different matrix decomposition methods, named LU decomposition and QR decomposition (Gram-Schmidt process, Modified Gram-Schmidt process, Householder transformations). In the future, I may apply matrix decomposition algorithm to neural networks. I hope it will be much more efficient than the regularisers methods.

## References

1. https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/04LinearAlgebra/lup/
2. https://math.unm.edu/~loring/links/linear_s08/LU.pdf
3. https://johnfoster.pge.utexas.edu/numerical-methods-book/
4. https://web.cs.ucdavis.edu/~bai/publications/andersonbaidongarra92.pdf
5. https://deepai.org/machine-learning-glossary-and-terms/qr-decomposition
6. https://en.wikipedia.org/wiki/Matrix_decomposition
7. http://homepages.math.uic.edu/~jan/mcs507f13/
8. https://www.cis.upenn.edu/~cis610/Gram-Schmidt-Bjorck.pdf
9. https://wikivisually.com/wiki/Gram%E2%80%93Schmidt_process
10. https://rpubs.com/aaronsc32/qr-decomposition-householder
11. https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
12. https://homel.vsb.cz/~dom033/predmety/parisLA/05_orthogonalization.pdf
13. https://core.ac.uk/download/pdf/82066579.pdf