---
title: Simpson Rule for Definite Integrals
top: false
cover: false
toc: true
mathjax: true
date: 2021-06-09 01:15:56
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/06/09/2021-06-09-simpson-rule-for-definite-integrals/wallhaven-g7jg63.png?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/06/09/2021-06-09-simpson-rule-for-definite-integrals/wallhaven-g7jg63.png?raw=true
summary: Simpson's rules are numerous approximations for definite integrals in numerical analysis, named after English mathematician Thomas Simpson (1710−1761). In calculus, basically, there are two ways to approximate the value of an integral, Reimann sums and Trapezoidal sums. However, calculating the value of an integral, we need to compute the areas of a zillion rectangles or more to get a better result. Therefore, we use Simpson's Rule, which is a way to approximate integrals without having to deal with lots of narrow rectangles.
categories: Mathematics
tags:
	- Numerical Analysis
	- Integrals
---

# Introduction

Simpson's rules are numerous approximations for definite integrals in numerical analysis, named after English mathematician Thomas Simpson (1710−1761). In calculus, basically, there are two ways to approximate the value of an integral, Reimann sums and Trapezoidal sums. However, calculating the value of an integral, we need to compute the areas of a zillion rectangles or more to get a better result. Therefore, we use Simpson's Rule, which is a way to approximate integrals without having to deal with lots of narrow rectangles.

# Simpson's 1/3 Rule

The most basic of these rules, called Simpson's 1/3 rule, or just Simpson's rule, reads

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\int_{a}^{b} f(x) dx \approx \frac{b-a}{6} [f(a) + 4 f(\frac{a+b}{2}) + f(b)]
	{% endmathjax %}
</div>

Introducing the step size {% mathjax %} h = \frac{b-a}{2} {% endmathjax %} this is also commonly written as

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\int_{a}^{b} f(x) dx \approx \frac{h}{3} [f(a) + 4 f(\frac{a+b}{2}) + f(b)]
	{% endmathjax %}
</div>

Because of the {% mathjax %} \frac{1}{3} {% endmathjax %} factor Simpson's rule is also referred to as Simpson's 1/3 rule.

# Simpson's 3/8 rule

Thomas Simpson proposed Simpson's 3/8 rule, often known as Simpson's second rule, as another approach for numerical integration. Rather than a quadratic interpolation, it uses a cubic interpolation. The 3/8 rule of Simpson is as follows:

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\int_{a}^{b} f(x) dx \approx \frac{3h}{8} [f(a) + 3 f(\frac{2a+b}{3}) + 3 f(\frac{a+2b}{3}) + f(b)]
	{% endmathjax %}
</div>

# Numerical Analysis

To obtain an approximation of the definite integral {% mathjax %} \int_{a}^{b} f(x) dx {% endmathjax %} using Simpson’s Rule, we partition the interval {% mathjax %} [a, b] {% endmathjax %} into an even number *n* of subintervals, each of width is {% mathjax %} \Delta x = \frac{b-a}{n} {% endmathjax %}. If the function {% mathjax %} f(x) {% endmathjax %} is continuous on {% mathjax %} [a, b] {% endmathjax %}, then 

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\int_{a}^{b} f(x) dx \approx \frac{\Delta x}{3} [f(x_0) + 4 f(x_1) + 2 f (x_2) + 4 f(x_3) + 2 f(x_4) + \ldots + 4 f(x_{n-1}) + f(x_{n})]
	{% endmathjax %}
</div>

The coefficients in Simpson’s Rule have the following pattern: {% mathjax %} 1, 4, 2, 4, 2, \ldots, 4, 2, 4, 1 {% endmathjax %} with {% mathjax %} n+1 {% endmathjax %} points.

## Example

The question is to use Simpson’s Rule with {% mathjax %} n = 4 {% endmathjax %} to approximate the integral {% mathjax %} \int_{0}^{8} \sqrt{x} dx {% endmathjax %}.

It is easy to see that the width of each subinterval is {% mathjax %} \Delta x = \frac{8-0}{4} = 2 {% endmathjax %} and the endpoints {% mathjax %} x_i = {0, 2, 4, 6, 8} {% endmathjax %}. Calculate the function values at the points {% mathjax %} x_i {% endmathjax %}, which is {% mathjax %} x_0 = \sqrt(0) {% endmathjax %}, {% mathjax %} x_1 = \sqrt(2) {% endmathjax %}, {% mathjax %} x_2 = \sqrt(4) {% endmathjax %}, {% mathjax %} x_3 = \sqrt(6) {% endmathjax %}, {% mathjax %} x_4 = \sqrt(8) {% endmathjax %}.

Substitute all these values into the Simpson’s Rule formula:

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\int_{0}^{8} \sqrt{x} dx \approx \frac{\Delta x}{3} [f(x_0) + 4 f(x_1) + 2 f (x_2) + 4 f(x_3) + f(x_4)] \approx 14.86
	{% endmathjax %}
</div>

The true solution for the integral is

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\int_{0}^{8} \sqrt{x} dx = \frac{32 \sqrt{2}}{3} \approx 15.08
	{% endmathjax %}
</div>

Hence, the error in approximating the integral is {% mathjax %} \epsilon = \frac{15.08-14.86}{15.08} \approx 0.015 = 1.5\% {% endmathjax %}

# Conclusion

Simpson's rule is a more accurate form of numerical integration than the Trapezoidal rule, and it should always be used before trying anything more complicated.

# References

1. https://web.stanford.edu/group/sisl/k12/optimization/MO-unit4-pdfs/4.2simpsonintegrals.pdf
2. https://www.math24.net/simpsons-rule
3. https://en.wikipedia.org/wiki/Simpson%27s_rule