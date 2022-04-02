---
title: Difference between Marginal Distribution and Conditional Distribution
top: false
cover: false
toc: true
mathjax: true
date: 2021-09-16 02:20:25
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/09/16/2021-09-16-difference-between-marginal-distribution-and-conditional-distribution/wallhaven-m92e8k.png?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/09/16/2021-09-16-difference-between-marginal-distribution-and-conditional-distribution/wallhaven-m92e8k.png?raw=true
summary: The possibility of two events occurring at the same time is known as joint probability. To better know the concept, we should take a closer look at marginal distribution and conditional distribution into detail.
categories: Statistics
tags:
  - Statistics
---

# Introduction

The possibility of two events occurring at the same time is known as joint probability. To better know the concept, we should take a closer look at marginal distribution and conditional distribution into detail.

# Main Difference

* Marginal Distribution: A marginal distribution considers a subset of variables. (What distribution would I get for {% mathjax %} X {% endmathjax %} if I ignored {% mathjax %} Y {% endmathjax %}?)
* Conditional Distribution: A conditional distribution fixes a subset of variables. (What distribution would I get for {% mathjax %} X {% endmathjax %} if I set {% mathjax %} Y=y {% endmathjax %}?)

To summarise what this is attempting to say, conditional distributions are concerned with determining probability for certain subsets of the population under consideration. They deal with determining the probability of one random variable given certain restrictions on the second random variable given two jointly distributed random variables. To put it in other words, a marginal distribution is the distribution of one random variable without any reference to the second random variable.

## Marginal Distribution

For {% mathjax %} P(X, Y) {% endmathjax %}, we have marginals {% mathjax %} P(X) {% endmathjax %} and {% mathjax %} P(Y) {% endmathjax %}. For discrete values, sum out the others:

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle P(X=x) = \sum_{y \in \mathcal{Y}} P(X=x, Y=y)
    {% endmathjax %}
</div>

For continuous values, integrate out the others:

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle p(x) = \int_{\mathcal{Y}} p(x, y) dy
    {% endmathjax %}
</div>

## Conditional Distribution

For both discrete and continuous, divide joint by marginal. The "given" is denoted using the pipe "|"" operator.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle P(X=x | Y=y) = \frac{P(X=x, Y=y)}{P(Y=y)} = \frac{P(X=x, Y=y)}{\sum_{x' \in \mathcal{X}} P(X=x', Y=y)}
    {% endmathjax %}
</div>

For continuous values, integrate out the others:

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle p(x | y) = \frac{p(x, y)}{p(y)} = \frac{p(x, y)}{\int_{\mathcal{X}} p(x', y) dx'}
    {% endmathjax %}
</div>

# Real-life Example

In a sample space with more than two events, if only the probability of a particular event occurring individually is considered, it is called the marginal probability. For example, 

* {% mathjax %} A_1 {% endmathjax %}: person who is from UoS
* {% mathjax %} A_2 {% endmathjax %}: person who is from NCHU
* {% mathjax %} B_1 {% endmathjax %}: person who is male
* {% mathjax %} B_2 {% endmathjax %}: person who is female

| | Male | Female | Sum |
| --- | --- | --- | --- |
| UoS | {% mathjax %} P(A_1 \wedge B_1) = 0.5 {% endmathjax %} | {% mathjax %} P(A_1 \wedge B_2) = 0.1 {% endmathjax %} | {% mathjax %} P(A_1) = 0.6 {% endmathjax %} |
| NCHU | {% mathjax %} P(A_2 \wedge B_1) = 0.1 {% endmathjax %} | {% mathjax %} P(A_2 \wedge B_2) = 0.3 {% endmathjax %} | {% mathjax %} P(A_2) = 0.4 {% endmathjax %} |
| Sum | {% mathjax %} P(B_1) = 0.6 {% endmathjax %} | {% mathjax %} P(B_2) = 0.4 {% endmathjax %} | {% mathjax %} 1.0 {% endmathjax %} |


where {% mathjax %} P(A_1) {% endmathjax %}, {% mathjax %} P(A_2) {% endmathjax %}, {% mathjax %} P(B_1) {% endmathjax %}, {% mathjax %} P(B_2) {% endmathjax %} are called marginal probabilities. As for conditional probability, for example, given the person is from NCHU, the probability of this person is a female is {% mathjax %} P(B_2 | A_2) = \frac{P(B_2 \wedge A_2)}{P(A_2)} = \frac{0.3}{0.4} = 0.75 {% endmathjax %}.

# Product Rule and Bayes' Rule

* Product Rule:

In probability theory, product rule also called chain rule, which allows you to calculate any member of a collection of random variables' joint distribution using just conditional probabilities. The rule is useful in the study of Bayesian networks, which are probabilistic networks that characterise a probability distribution using conditional probabilities.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle P(X=x, Y=y) = P(X=x | Y=y) P(Y=y) = P(Y=y | X=x) P(X=x)
    {% endmathjax %}
</div>

* Bayes' Rule

In probability theory and statistics, Bayes' rule describes the probability of an event, based on prior knowledge of conditions that might be related to the event.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle P(X=x | Y=y) = \frac{P(Y=y | X=x) P(X=x)}{P(Y=y)}
    {% endmathjax %}
</div>

# Conclusions

The marginal probability is the probability of a single event occurring, independent of other events. A conditional probability, on the other hand, is the probability that an event occurs given that another specific event has already occurred. This means that the calculation for one variable is dependent on another variable.

# References

1. https://www.youtube.com/watch?v=CQS4xxz-2s4
2. https://en.wikipedia.org/wiki/Marginal_distribution
3. https://murphymind.blogspot.com/2011/10/probability.html