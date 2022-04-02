---
title: Jensen's Inequality and its Role in Finance
top: false
cover: false
toc: true
mathjax: true
date: 2021-05-26 22:34:54
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/26/2021-05-26-jensen-inequality-and-its-role-in-finance/wallhaven-x8ey2d.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/26/2021-05-26-jensen-inequality-and-its-role-in-finance/wallhaven-x8ey2d.jpg?raw=true
summary: Jensen's inequality is perhaps the most famous theorem in quantitative finance (note that it is a "theorem" and not a model or a formula) and it is the reason why financial derivatives have value. Concept of convexity, Jensen's inequality, randomness and volatility of an asset price are intricately linked.
categories: Statistics
tags:
	- Mathematics
	- Statistics
	- Finance
---

# Introduction

Jensen's inequality is perhaps the most famous theorem in quantitative finance (note that it is a "theorem" and not a model or a formula) and it is the reason why financial derivatives have value. Concept of convexity, Jensen's inequality, randomness and volatility of an asset price are intricately linked.

# Definition

Jensen's Inequality states that if {% mathjax %} f {% endmathjax %} is a convex function and {% mathjax %} X {% endmathjax %} is a random variable then

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    E[f(x)] > f(E[X])
    {% endmathjax %}
</div>

# Example

You roll a die, and square the number of spots you get, finally you win that many dollars. For this exercise {% mathjax %} f(x) {% endmathjax %} is {% mathjax %} x^2 {% endmathjax %}, which is a convex function. So {% mathjax %} E[f(x)] {% endmathjax %} is {% mathjax %}1 + 4 + 9 + 16 + 25 + 36 = 91  {% endmathjax %} divided by 6, so {% mathjax %} \frac{97}{6} {% endmathjax %}. But {% mathjax %} E[x] {% endmathjax %} is {% mathjax %} \frac{7}{2} {% endmathjax %}, so {% mathjax %} f(E[x]) {% endmathjax %} is {% mathjax %} \frac{49}{4} {% endmathjax %}.

# Conclusion

Jensen’s Inequality is a useful tool in mathematics, specifically in applied fields such as probability and statistics. Jensen's Inequality and convexity can also be used to explain the relationship between randomness in stock prices and the value inherent in options, the latter typically having some convexity. A common application of the inequality is in the comparison of arithmetic and geometric means when averaging the financial returns for a time interval.

# References

1. https://ebrary.net/7091/business_finance/what_jensens_inequality_and_what_its_role_finance
2. https://www.risklatte.xyz/Articles/QuantitativeFinance/QF187.php
3. Wilmott, Paul Wilmott on Quantitative Finance (2006)
4. https://machinelearningmastery.com/a-gentle-introduction-to-jensens-inequality/