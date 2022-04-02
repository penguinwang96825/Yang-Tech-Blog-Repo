---
title: A Proof that e is Irrational
top: false
cover: false
toc: true
mathjax: true
date: 2021-03-17 21:51:49
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/17/2021-03-17-prove-that-e-is-irrational/wallhaven-odq5ml.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/17/2021-03-17-prove-that-e-is-irrational/wallhaven-odq5ml.jpg?raw=true
summary: In this article, I'll try and show that e, sometimes called Euler's number, is an irrational number 2.718281828459045.... Euler's number is a fantastic number, and it plays a role in just about every aspect of physics, maths, and statistics. There are many ways of calculating the value of e, but none of them ever give a totally exact answer, because e is irrational and its digits go on forever without repeating.
tags:
    - Mathematics
    - Euler's Number
categories: Mathematics
---


# Introduction

In this article, I'll try and show that e, sometimes called Euler's number, is an irrational number 2.718281828459045... and so on. Euler's number is a fantastic number, and it plays a role in just about every aspect of physics, maths, and statistics. There are many ways of calculating the value of e, but none of them ever give a totally exact answer, because e is irrational and its digits go on forever without repeating. 

# Beautiful Property

This exponential function, {% mathjax %} e^x {% endmathjax %}, is the only one that if we differentiate it, we get the same function {% mathjax %} e^x {% endmathjax %}.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \frac{d}{dx} e^x = e^x
    {% endmathjax %}
</div>

The next differential of it we get 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \frac{d^2}{dx^2} e^x = e^x
    {% endmathjax %}
</div>

and it goes on an on and on, and that's the beauty of this particular function that all its derivatives and slopes match the actual function itself. It means it carries on forever, because the slope just keeps the same no matter how far it goes, whereas other functions will die off if we differentiate them.

# A Proof that e is Irrational

In this section, we want to prove that e can't be represented as the ratio of two integers, that's what an irrational number is.

We assume that e can be represented as the ratio of two integers, and write {% mathjax %} e = \frac{p}{q} {% endmathjax %} with p and q integers. By cancelling, we may assume that p and q are not both even (if they are, we can simply keep cancelling powers of 2 until one of them is not). If our original assumption (that e is rational) is not correct, then we know that e is irrational.

One thing to mention is that e has a nice infinite series expansion, it's

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    e = 1 + \frac{1}{1!} + \frac{1}{2!} + \frac{1}{3!} + \ldots
    {% endmathjax %}
</div>

where in mathematics, the factorial of a positive integer n, denoted by n!, is the product of all positive integers less than or equal to n.

To simplify, we call {% mathjax %} e = \frac{p}{q} {% endmathjax %} function 1 and {% mathjax %} e = 1 + \frac{1}{1!} + \frac{1}{2!} + \frac{1}{3!} + \ldots {% endmathjax %} function 2. And we're going to multiply this function 2 by q factorial. Then we will get function 3

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    q! \times e = q! + \frac{q!}{1!} + \frac{q!}{2!} + \frac{q!}{3!} + \ldots + \frac{q!}{q!} + \ldots
    {% endmathjax %}
</div>

From our assumption, we know that {% mathjax %} e = \frac{p}{q} {% endmathjax %}, that means 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    q! \times e = (q-1)! \times qe = (q-1)! \times p
    {% endmathjax %}
</div>

So we know that {% mathjax %} q! e {% endmathjax %} must be an integer. And also we know that {% mathjax %} q! + \frac{q!}{1!} + \frac{q!}{2!} + \frac{q!}{3!} + \ldots + \frac{q!}{q!} {% endmathjax %} is an integer. Then the rest of function 3 (components after {% mathjax %} \frac{q!}{q!} {% endmathjax %}) should be an integer right?

Let's derive the rest of the function 3 after {% mathjax %} \frac{q!}{q!} {% endmathjax %}, 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    R = q! (\frac{1}{(q+1)!} + \frac{1}{(q+2)!} + \frac{1}{(q+3)!} + \ldots)
    {% endmathjax %}
</div>

and we can get 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    R = \frac{1}{q+1} (1 + \frac{1}{q+2} + \frac{1}{(q+2)(q+3)} + \ldots)
    {% endmathjax %}
</div>

which is clearly 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    R < \frac{1}{q+1} (1 + \frac{1}{q+1} + \frac{1}{(q+1)^2} + \ldots)
    {% endmathjax %}
</div>

According to Gauss, if {% mathjax %} x<1 {% endmathjax %}, then {% mathjax %} \displaystyle \lim_{n\to\infty} S_n = 1 + x + x^2 + x^3 + \ldots + x^n = \frac{1}{1-x} {% endmathjax %}. In this case, {% mathjax %} x = \frac{1}{q+1} {% endmathjax %}, so we have {% mathjax %} S_{\infty} = \frac{1}{1 - \frac{1}{q+1}} {% endmathjax %}. Finally, rearranging this then we can get {% mathjax %} S_{\infty} = \frac{q+1}{q} {% endmathjax %}.

Let's come back to the remainder R, the rest, we said it must be an integer.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    R < \frac{1}{q+1} \times \frac{q+1}{q} = \frac{1}{q}
    {% endmathjax %}
</div>

Hence we know R is bounded between 0 and {% mathjax %} \frac{1}{q} {% endmathjax %} and q is bigger than 1, so this is a fraction less than 1. It means R can not be an integer. It contradicts! Therefore e can not be represented as a rational number, so it has to be irrational.

# Conclusion

This is the most well-known proof by Joseph Fourier using contradiction. Hope you like it!

## References

1. https://www2.math.upenn.edu/~kazdan/202F13/hw/e-irrat.pdf
2. https://math.stackexchange.com/questions/713467/e-is-irrational