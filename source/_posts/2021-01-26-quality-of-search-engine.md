---
title: Quality of Search Engine - PageRank
date: 2021-01-26 09:25:00
author: Yang Wang
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/01/26/2021-01-26-quality-of-search-engine/hoover-tung.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/01/26/2021-01-26-quality-of-search-engine/southpark.jpg?raw=true
summary: Web search engines such as Google.com distinguish themselves by the qualtiy of their returns to search queries. I will discuss a rough approximation of Google's method for judging the qualtiy of web pages by using PageRank.
categories: Mathematics
tags:
  - MATLAB
  - PageRank
---

Web search engines such as Google.com distinguish themselves by the qualtiy of their returns to search queries. I will discuss a rough approximation of Google's method for judging the qualtiy of web pages by using PageRank.

# Introduction

When a web search is initiated, there is rather a complex series of tasks that are carried out by the search engine. One obvious task is **word-matching**, to find pages that contain the query words, in the title or body of the page. Another key task is to **rate the pages** that are identified by the first task, to help the user wade through the possibly large set of choices. This is basically how information retrieval task works!

# PageRank

When you google for a keyword "steak sauce", it would possibly return several million pages, begining with the some recipes for steak, a reasonably outcome. How is this ranking determined? The answer to this question is that Google.com assigns a non-negative real number, called the `page rank`, to each web page that it indexes.

Consider a graph like the following: 

{% asset_img figure.png %}

Each of $n$ nodes represents a web page, and a directed edge from node {% mathjax %} i {% endmathjax %} to {% mathjax %} j {% endmathjax %} means that page {% mathjax %} i {% endmathjax %} contains a web link to page {% mathjax %} j {% endmathjax %}. Let {% mathjax %} A {% endmathjax %} denote the adjacency matrix, an {% mathjax %} n \times n {% endmathjax %} matrix whose {% mathjax %} ij {% endmathjax %}-th entry is 1 if there is a link from node {% mathjax %} i {% endmathjax %} to node {% mathjax %} j {% endmathjax %}, and 0 otherwise.

<div style="display: flex;justify-content: center;">
  {% mathjax %}
   A = 
  \left[ {\begin{array}{ccccccccccccccc}
   0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
   0 & 0 & 1 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
   0 & 1 & 0 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
   0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
   1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 \\ 
   0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
   0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 1 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 1 & 0 & 1 \\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 1 & 0 \\
  \end{array} } \right]
  {% endmathjax %}
</div>

The invention of Google imagined a surfer on a network of {% mathjax %} n {% endmathjax %} pages, who currently sits at page {% mathjax %} i {% endmathjax %} with probability {% mathjax %} p_i {% endmathjax %}. Next, the surfer either moves to a random page (with fixed probability {% mathjax %} q {% endmathjax %}, often around 0.15) or with probability {% mathjax %} 1-q {% endmathjax %}, clicks randomly on a link from the current page {% mathjax %} i {% endmathjax %}. The probability that the surfer moves from page {% mathjax %} i {% endmathjax %} to page {% mathjax %} j {% endmathjax %} after the click is 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    q \cdot \frac{1}{n} + (1-q) \cdot \frac{A_{ij}}{n_i}
    {% endmathjax %}
</div>

where {% mathjax %} A_{ij} {% endmathjax %} is the entry of the adjacency matrix {% mathjax %} A {% endmathjax %}, and {% mathjax %} n_i {% endmathjax %} is the sum of the {% mathjax %} i {% endmathjax %}-th row of {% mathjax %} A {% endmathjax %}. Since the time is arbitrary, the probability of being at node {% mathjax %} j {% endmathjax %} is the sum of this expression over all {% mathjax %} i {% endmathjax %}, and it is independent of time; that is, 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    p_j = \sum_{i}{} (q \cdot \frac{p_i}{n} + (1-q) \cdot \frac{p_i}{n_i} \cdot A_{ij})
    {% endmathjax %}
</div>

which is equivalent in matrix terms to the eigenvalue equation

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    p = Gp 
    {% endmathjax %}
</div>

where

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    G(i, j) = \frac{q}{n}+\frac{A_{ji} (1-q)}{\sum_{i=1}^{n} A_{ji}}
    {% endmathjax %}
</div>

Let's contruct the adjacency matrix using MATLAB!

```matlab
%% Construct adjacency matrix A, there are 34 arrows
clear all; clc
n = 15
i = [5 1 3 2 4 8 2 9 3 9 2 12 3 12 1 13 5 6 7 9 14 6 7 8 12 14 4 15 10 14 13 15 11 14];
j = [1 2 2 3 3 4 5 5 6 6 7 7 8 8 9 9 10 10 10 10 10 11 11 11 11 11 12 12 13 13 14 14 15 15];
A = sparse(i, j, 1, n, n);
full(A);
```

We also know that {% mathjax %} q {% endmathjax %}, {% mathjax %} n {% endmathjax %}, {% mathjax %} A_{ij} {% endmathjax %} and {% mathjax %} (1-q) {% endmathjax %} are non-negative, so {% mathjax %} G(i, j)>0 {% endmathjax %}. Since the total of transition probability from a state {% mathjax %} i {% endmathjax %} to all other states must be1, that is, by the construction of the sum of all non-negative elements inside each matrix column is equal to unity, thus this Google matrix $G$ is a stochastic matrix.

> In mathematics, a stocastic matrix is a square matrix used to describe the transitions of a Markov chain. Each of the entries is a non-negative real number representing a probability. It is also called a probability matrix, transition matrix, or Markov matrix.

Next step is to contruct the matrix {% mathjax %} G {% endmathjax %} for the network, and verify the given dominant eigenvector {% mathjax %} p {% endmathjax %}.

```matlab
%% Construct google matrix G
G = zeros(n, n)
q = 0.15
for i = 1:n
    for j = 1:n
        G(i, j) = q/n + A(j, i) * (1-q) / sum(A(j, :));
    end
end
```

We also define a power method to computeeigen values and eigenvectors.

```matlab
function [V, D] = power(A, iter)
    % Power Method
    [m, n] = size(A);
    % Random number generator
    rng(1)
    % Initial vector
    x0 = randn(n, 1);
    x = x0;
    for j = 1:iter
        u = x / norm(x);
        x = A * u;
        % Rayleigh Quotient
        lam = u' * x;
        L(j) = lam;
    end
    % Eigenvalue D
    D = L(end);
    u = x / norm(x);
    % Eigenvector V
    V = u;
```

We use the above function to find the dominant eigenvector {% mathjax %} p {% endmathjax %}.

```matlab
iter = 1000;
[p, D] = power(G, iter);
p = p / sum(p);
```

The entries of {% mathjax %} p {% endmathjax %} represents the importance level of each 15 websites.

```matlab
%---------- Results ----------%
p = 
    0.0268
    0.0299
    0.0299
    0.0268
    0.0396
    0.0396
    0.0396
    0.0396
    0.0746
    0.1063
    0.1063
    0.0746
    0.1251
    0.1163
    0.1251
```

## Jump Probability

What is the purpose of the jump probability? There is some probability {% mathjax %} q {% endmathjax %} such that at every step the walk has an {% mathjax %} q {% endmathjax %} probability of jumping to a uniformly chosen random page. They tell us that {% mathjax %} q {% endmathjax %} is set to some moderately small constant like 0.15. This is equivalent to adding a low-weight edge between every pair of vertices.

You may wonder what if we change the jump probability {% mathjax %} q {% endmathjax %}? I describe the resulting changes below.

| page         | q = 0.15          | q = 0          | q = 0.5          |
| ------------ |:-----------------:| --------------:| ----------------:|
| 1            | 0.0268            | 0.0154         | 0.0467           |
| 2            | 0.0299            | 0.0116         | 0.0540           |
| 3            | 0.0299            | 0.0116         | 0.0540           |
| 4            | 0.0268            | 0.0154         | 0.0467           |
| 5            | 0.0396            | 0.0309         | 0.0536           |
| 6            | 0.0396            | 0.0309         | 0.0536           |
| 7            | 0.0396            | 0.0309         | 0.0536           |
| 8            | 0.0396            | 0.0309         | 0.0536           |
| 9            | 0.0746            | 0.0811         | 0.0676           |
| 10           | 0.1063            | 0.1100         | 0.0946           |
| 11           | 0.1063            | 0.1100         | 0.0946           |
| 12           | 0.0746            | 0.0811         | 0.0676           |
| 13           | 0.1251            | 0.1467         | 0.0905           |
| 14           | 0.1163            | 0.1467         | 0.0786           |
| 15           | 0.1163            | 0.1467         | 0.0905           |

{% asset_img jump.png %}

How does PageRank work for disjoint connected components? PageRank "fixes" this with the "teleportation" parameter, famously set to 0.15, that weakly connects all nodes to all other nodes. This can be thought of as an implicit link to all {% mathjax %} N {% endmathjax %} other nodes in the graph with weight {% mathjax %} \frac{0.15}{N} {% endmathjax %}. Making this teleportation factor higher should allow a higher probability of escaping from a small connected component.

Without transportation, PageRank doesn't provide a way to compute the centralities of nodes in different components. That is, computing the PageRank of a node tells you its relative importance "within that component", but doen't tell anything about how different components compare.

Suppose that Page 7 in the network wanted to improve its page rank, compared with its competitor Page 6, by persuading Pages 2 and 12 to more prominently display its links to Page 7. Model this by replacing {% mathjax %} A_{2, 7} {% endmathjax %} and {% mathjax %} A_{12,7} {% endmathjax %} by 2 in the adjacency matrix. Let's see whether this strategy succeed or not.

```matlab
%% Construct adjacency matrix A
n = 15
i = [5 1 3 2 4 8 2 9 3 9 2 12 3 12 1 13 5 6 7 9 14 6 7 8 12 14 4 15 10 14 13 15 11 14];
j = [1 2 2 3 3 4 5 5 6 6 7 7 8 8 9 9 10 10 10 10 10 11 11 11 11 11 12 12 13 13 14 14 15 15];
A = sparse(i, j, 1, n, n);
A(2, 7) = 2;
A(12, 7) = 2;
full(A);

%% Construct Google Matrix
G = zeros(0)
q = 0.15
for i = 1:n
    for j = 1:n
        G(i, j) = q/n + A(j, i) * (1-q) / sum(A(j, :));
    end
end

%% Solve eigenvalue problem for the ranking vector
iter = 1000;
[p, D] = power(G, iter);
p = p / sum(p);
bar(p);
title('PageRank with q = 0.15');
```

{% asset_img page7.png %}

As you can see, the rank of Page 7 successfully exceed Page 6.

--- 

Let's study the effect of removing Page 10 from the network (All links from Page 10 are deleted). Which page ranks increase, and which decrease?

{% asset_img page10.png %}

| page         | q = 0.15          | q = 0.15 (Remove Page 10)  | Increase or Decrease          |
| ------------ |:-----------------:| --------------------------:| -----------------------------:|
| 1            | 0.0268            | 0.0462                     | Increase                      |
| 2            | 0.0299            | 0.0393                     | Increase                      |
| 3            | 0.0299            | 0.0341                     | Increase                      |
| 4            | 0.0268            | 0.0305                     | Increase                      |
| 5            | 0.0396            | 0.0426                     | Increase                      |
| 6            | 0.0396            | 0.0412                     | Increase                      |
| 7            | 0.0396            | 0.0496                     | Increase                      |
| 8            | 0.0396            | 0.0481                     | Increase                      |
| 9            | 0.0746            | 0.0506                     | Decrease                      |
| 10           | 0.1063            | 0.0100                     | Decrease                      |
| 11           | 0.1063            | 0.1669                     | Increase                      |
| 12           | 0.0746            | 0.1005                     | Increase                      |
| 13           | 0.1251            | 0.0492                     | Decrease                      |
| 14           | 0.1163            | 0.1085                     | Decrease                      |
| 15           | 0.1163            | 0.1826                     | Increase                      |

There are 73% of the pages will increase, and 27% od pages will decrease. The reason is that, before removing Page 10, it can be seen that there are many pages (33% of the pages) point to Page 10, it means that Page 10 is relatively important to other websites.

# Conclusion

PageRank is a powerful algorithm and has a wide application, such as ranking the football teams, ranking tweets in Twitter, or ranking track athletes. The merit of PageRank comes from its power in evaluating *network measures* in a connected system. Clearly, the surprisingly wide variety of these existing applications of PageRank point to a rich future for the algorithm in research contexts of all types. It seems intuitive that any problem in any field where a network comes into play might benefit from using PageRank.

## References

1. [[link](https://www.sciencedirect.com/science/article/abs/pii/S016975529800110X)] S. Brin and L. Page, The anatomy of a large-scale hypertextual web search engine
2. [[link](https://projecteuclid.org/download/pdf_1/euclid.involve/1513733537)] Laurie Zack, Ron Lamb and Sarah Ball, An Application of Google's PageRank to NFL Rankings
3. [[link](https://web.stanford.edu/class/msande233/handouts/lecture8.pdf)] Ashish Goel, Applications of PageRank to Recommendation Systems
4. [[link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178458)] Clive B. Beggs, A Novel Application of PageRank and User Preference Algorithms for Assessing the Relative Performance of Track Athletes in Competition
5. [[link](https://www.hindawi.com/journals/ijta/2019/8612021/)] Shahram Payandeh and Eddie Chiu, Application of Modified PageRank Algorithm for Anomaly Detection in Movements of Older Adults