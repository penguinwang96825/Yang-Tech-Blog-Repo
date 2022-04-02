---
title: Missionaries and Cannibals Problem
date: 2021-01-27 09:25:00
author: Yang Wang
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/01/27/2021-01-27-missionaries-and-cannibals-problem/janis-beitins.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/01/27/2021-01-27-missionaries-and-cannibals-problem/koningsberg-bridge-problem.jpg?raw=true
summary: This is actually one of my project when I was in college during my "Mathematical Programming" class. I used R programming language to solve missionaries and cannibals problem with matrix analysis and graph theory.
top: true
categories: Mathematics
tags:
  - R
  - Linear Algebra
---

This is actually one of my project when I was in college during my "Mathematical Programming" class. I used R programming language to solve missionaries and cannibals problem with matrix analysis and graph theory.

# Introduction

The missionaries and cannibals problem, and the closely related jealous husbands problem, are classic river-crossing logic puzzles. In the missionaries and cannibals problem, three missionaries and three cannibals must cross a river using a boat which can carry at most two people, under the constraint that, for both banks, if there are missionaries present on the bank, they cannot be outnumbered by cannibals (if they were, the cannibals would eat the missionaries). The boat cannot cross the river by itself with no people on board.

{% asset_img mc-search-space.png %}

# Solving

A system for solving the Missionaries and Cannibals problem whereby the current state is represented by a simple vector {% mathjax %} (m, c, b) {% endmathjax %}. The vector's elements represent the number of missionaries, cannibals, and whether the boat is on the the east side or on the west side, respectively. Suppose the boat and all of the missionaries and cannibals start on the east side, the vector is initialized to {% mathjax %} (3, 3, e) {% endmathjax %}. According to the problem, {% mathjax %} m = 0, 1, 2, 3 {% endmathjax %} and {% mathjax %} c = 0, 1, 2, 3 {% endmathjax %}, therefore, there are {% mathjax %} 4\times4\times2=32 {% endmathjax %} different status. However, the nummber of missionaries must be greater than the number of cannibals, and there should always be people on the shoreside while the boat is on that side, so the following states do not exist:

<div style="display: flex;justify-content: center;">
  {% mathjax %}
  \displaylines{(0, 0, e), (1, 2, e), (1, 2, w), (1, 3, e), (1, 3, w), (2, 3, e), (2, 3, w) \\
                (0, 0, w), (2, 1, e), (2, 1, w), (2, 0, e), (2, 0, w), (1, 0, e), (1, 0, w) }
  {% endmathjax %}
</div>

There are only 18 states left, and we use {% mathjax %} v_1, v_2, \ldots, v_{18} {% endmathjax %} to represent a set of vertexs as following:

<div style="display: flex;justify-content: center;">
    <span>
        {% mathjax %}
        \displaylines{v_1=(3, 3, e) \quad v_{10}=(3, 3, w) \\
                      v_2=(3, 2, e) \quad v_{11}=(3, 2, w) \\
                      v_3=(3, 1, e) \quad v_{12}=(3, 1, w) \\
                      v_4=(3, 0, e) \quad v_{13}=(3, 0, w) \\
                      v_5=(2, 2, e) \quad v_{14}=(2, 2, w) \\
                      v_6=(2, 3, e) \quad v_{15}=(1, 1, w) \\
                      v_7=(0, 3, e) \quad v_{16}=(0, 3, w) \\
                      v_8=(0, 2, e) \quad v_{17}=(0, 2, w) \\
                      v_9=(0, 1, e) \quad v_{18}=(0, 1, w) }
        {% endmathjax %}
    </span>
</div>

We know that everyone departs from east side, that is, the initial vertex is {% mathjax %} v_1 {% endmathjax %}. Suppose 1 missionary and 1 cannibal tend to go from east side to west side, then it should be represented as {% mathjax %} v_1 {% endmathjax %} to {% mathjax %} v_{15} {% endmathjax %}, and we will say {% mathjax %} v_1 {% endmathjax %} and {% mathjax %} v_{15} {% endmathjax %} are adjacent to each other: 

<div style="display: flex;justify-content: center;">
  {% mathjax %} v_1\xrightarrow[\text{}] {\text{m, c} }v_{15} {% endmathjax %}
</div>

Obviously, the procedure is reversible:

<div style="display: flex;justify-content: center;">
  {% mathjax %} v_{15}\xrightarrow[\text{}] {\text{m, c} }v_{1} {% endmathjax %}
</div>

This means the two vertexs has symmetric relation, we then use undirected arrow to connect them. For example, let's say we make missionaries and cannibals start from {% mathjax %} v_1 {% endmathjax %}, then there will be three possible conditions:

<div style="display: flex;justify-content: center;">
    <span>
        {% mathjax %}
        \displaylines{v_{1}\xrightarrow[\text{}]{ \text{m, c} }v_{15} \\
                      v_{1}\xrightarrow[\text{}]{ \text{c, c} }v_{17} \\
                      v_{1}\xrightarrow[\text{}]{ \text{c} }v_{18} }
        {% endmathjax %}
    </span>
</div>

By repeating the above algorithm, we can get an undirected graph.

{% asset_img undirected-graph-of-river-crossing-problem1.jpg %}

In order to compute this in matrix, I define a {% mathjax %} 18\times18 {% endmathjax %} adjacency matrix {% mathjax %} A=[a_{ij}] {% endmathjax %} to represent the undirected graph: if {% mathjax %} v_i {% endmathjax %} and {% mathjax %} v_j {% endmathjax %} are adjacent to each other, then 1, otherwise 0. Moreover, for every {% mathjax %} i {% endmathjax %}, we let the value be 0, that is, {% mathjax %} a_{ii}=0 {% endmathjax %}.

<div style="display: flex;justify-content: center;">
  {% mathjax %}
   A = 
  \left[ {\begin{array}{cc}
   0_{6\times6} & B_{9\times9} \\
   B_{9\times9} & 0_{6\times6} \\
  \end{array} } \right]
  {% endmathjax %}
</div>

where {% mathjax %} B=[b_{ij}] {% endmathjax %} is a {% mathjax %} 9\times9 {% endmathjax %} symmetric matrix.

<div style="display: flex;justify-content: center;">
  {% mathjax %}
   B = 
  \left[ {\begin{array}{ccccccccccccccc}
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 1 & 1 \\
   0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 0 \\
   0 & 0 & 0 & 0 & 1 & 0 & 1 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
   0 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 \\ 
   1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
   0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
   1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
   1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
  \end{array} } \right]
  {% endmathjax %}
</div>

From above, we can tell that {% mathjax %} b_{1, 6}=1 {% endmathjax %} illustrate {% mathjax %} v_1 {% endmathjax %} and {% mathjax %} v_{15} {% endmathjax %} are adjacent to each other. We define the `journey` starts from {% mathjax %} v_i {% endmathjax %} to {% mathjax %} v_j {% endmathjax %} is a series of paths:

<div style="display: flex;justify-content: center;">
  {% mathjax %}
  v_i=u_0 \rightarrow u_1 \rightarrow \ldots \rightarrow u_k=v_j
  {% endmathjax %}
</div>

If we do not pass through the vertex of repeat visits, which means that {% mathjax %} u_k {% endmathjax %} are different from each other, then we call it is a path of length {% mathjax %} k {% endmathjax %}. Our goal is to start at {% mathjax %} v_1 {% endmathjax %} and end at {% mathjax %} v_{10} {% endmathjax %}, because {% mathjax %} v_1 {% endmathjax %} part of the east side, and {% mathjax %} v_{10} {% endmathjax %} belongs to the west side, so the length of the path {% mathjax %} k {% endmathjax %} must be an odd number. For odd numbers {% mathjax %} k {% endmathjax %}, it is not difficult to confirm:

<div style="display: flex;justify-content: center;">
  {% mathjax %}
   A^k = 
  \left[ {\begin{array}{cc}
   \displaylines{0_{6\times6} & B_{9\times9}^k \\
                 B_{9\times9}^k & 0_{6\times6} \\}
  \end{array} } \right]
  {% endmathjax %}
</div>

The power of the adjacency matrix {% mathjax %} A {% endmathjax %}, represented as {% mathjax %} A^k_{ij} {% endmathjax %}, gives the number of paths of length k between vertices {% mathjax %} v_i {% endmathjax %} and {% mathjax %} v_j {% endmathjax %}. ([mathematical proof](https://mathworld.wolfram.com/GraphPower.html))

This property provides an easy method to find the shortest path: calculated continuously {% mathjax %} A^k {% endmathjax %}, and {% mathjax %} k=1, 3, 5 \ldots {% endmathjax %} until {% mathjax %} A^d(1, 10) {% endmathjax %} is greater than zero, that is, {% mathjax %} B^d(1, 1)>0 {% endmathjax %} which means that we have arrived at the destination {% mathjax %} v_{10} {% endmathjax %}, but also know the shortest path length is equal to {% mathjax %} d {% endmathjax %}, and the total number of path is equal to {% mathjax %} B^d(1, 1) {% endmathjax %}.

# Computation in R

I define a variable B representing the adjacency matrix mentioned above and build a function to compute the power to a matrix.

```r
B <- matrix(c(
    0, 0, 0, 0, 0, 1, 0, 1, 1, 
    0, 0, 0, 0, 0, 1, 1, 1, 0, 
    0, 0, 0, 0, 1, 0, 1, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 1, 0, 1, 0, 0, 0, 0,  
    1, 1, 0, 0, 0, 0, 0, 0, 0,  
    0, 1, 1, 0, 0, 0, 0, 0, 0,  
    1, 1, 0, 0, 0, 0, 0, 0, 0, 
    1, 0, 0, 0, 0, 0, 0, 0, 0), nrow=9)

power = function(x, n) Reduce(`%*%`, replicate(n, x, simplify=FALSE))
```

---

For {% mathjax %} k=3 {% endmathjax %}:

```r
power(B, 3)

     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
[1,]    0    0    0    0    0    5    2    5    3
[2,]    0    0    0    0    1    5    4    5    2
[3,]    0    0    1    0    3    1    3    1    0
[4,]    0    0    0    0    0    0    0    0    0
[5,]    0    1    3    0    3    0    1    0    0
[6,]    5    5    1    0    0    0    0    0    0
[7,]    2    4    3    0    1    0    0    0    0
[8,]    5    5    1    0    0    0    0    0    0
[9,]    3    2    0    0    0    0    0    0    0
```

Vertex first visited: {% mathjax %} v_{7+9}=v_{16} {% endmathjax %}

Shortest path length: {% mathjax %} d(1, 16)=3 {% endmathjax %}

Total number of path: 2

---

For {% mathjax %} k=5 {% endmathjax %}:

```r
power(B, 5)

     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
[1,]    0    0    0    0    2   25   14   25   13
[2,]    0    0    1    0    6   26   19   26   12
[3,]    0    1    5    0   10    7   11    7    2
[4,]    0    0    0    0    0    0    0    0    0
[5,]    2    6   10    0   10    1    5    1    0
[6,]   25   26    7    0    1    0    0    0    0
[7,]   14   19   11    0    5    0    1    0    0
[8,]   25   26    7    0    1    0    0    0    0
[9,]   13   12    2    0    0    0    0    0    0
```

Vertex first visited: {% mathjax %} v_{5+9}=v_{14} {% endmathjax %}

Shortest path length: {% mathjax %} d(1, 14)=5 {% endmathjax %}

Total number of path: 2

---

For {% mathjax %} k=7 {% endmathjax %}:

```r
power(B, 7)

     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
[1,]    0    0    2    0   18  127   80  127   63
[2,]    0    1    8    0   32  135   96  135   64
[3,]    2    8   21    0   36   41   46   41   16
[4,]    0    0    0    0    0    0    0    0    0
[5,]   18   32   36    0   35    9   22    9    2
[6,]  127  135   41    0    9    0    1    0    0
[7,]   80   96   46    0   22    1    7    1    0
[8,]  127  135   41    0    9    0    1    0    0
[9,]   63   64   16    0    2    0    0    0    0
```

Vertex first visited: {% mathjax %} v_{3+9}=v_{12} {% endmathjax %}

Shortest path length: {% mathjax %} d(1, 12)=7 {% endmathjax %}

Total number of path: 2

---

For {% mathjax %} k=9 {% endmathjax %}:

```r
power(B, 9)

     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
[1,]    0    2   22    0  118  651  432  651  317
[2,]    2   11   49    0  168  700  494  700  334
[3,]   22   49   86    0  139  226  210  226   98
[4,]    0    0    0    0    0    0    0    0    0
[5,]  118  168  139    0  128   60   97   60   20
[6,]  651  700  226    0   60    1   11    1    0
[7,]  432  494  210    0   97   11   38   11    2
[8,]  651  700  226    0   60    1   11    1    0
[9,]  317  334   98    0   20    0    2    0    0
```

Vertex first visited: {% mathjax %} v_{2+9}=v_{11} {% endmathjax %}

Shortest path length: {% mathjax %} d(1, 11)=9 {% endmathjax %}

Total number of path: 2

---

For {% mathjax %} k=11 {% endmathjax %}:

```r
power(B, 11)

     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
[1,]    4   28  164    0  690 3353 2284 3353 1619
[2,]   28   86  277    0  879 3628 2556 3628 1734
[3,]  164  277  360    0  574 1212 1011 1212  550
[4,]    0    0    0    0    0    0    0    0    0
[5,]  690  879  574    0  492  357  442  357  140
[6,] 3353 3628 1212    0  357   15   84   15    2
[7,] 2284 2556 1011    0  442   84  195   84   24
[8,] 3353 3628 1212    0  357   15   84   15    2
[9,] 1619 1734  550    0  140    2   24    2    0
```

Vertex first visited: {% mathjax %} v_{1+9}=v_{10} {% endmathjax %}

Shortest path length: {% mathjax %} d(1, 10)=11 {% endmathjax %}

Total number of path: 4

---

Therefore all the missionaries and cannibals have crossed the river safely.

# Conclusion

As a math major student, it may very well be true that we won't use some of the more abstract mathematical concepts we learn in school unless we choose to work in specific project. We are taught lots of useless things in math major. However, the underlying skills we developed in mathametics class, such as thinking logically and solving problems, will last a lifetime and help us solve work-related and real-world problems.

## References

1. https://ccjou.wordpress.com/2012/08/09/%E6%B8%A1%E6%B2%B3%E5%95%8F%E9%A1%8C/
2. https://stackoverflow.com/questions/3274818/matrix-power-in-r
3. http://page.mi.fu-berlin.de/rote/Papers/pdf/Crossing+the+bridge+at+night.pdf