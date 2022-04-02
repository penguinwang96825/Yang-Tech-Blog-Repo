---
title: Estimate π on π Day
top: false
cover: false
toc: true
mathjax: true
date: 2021-03-14 08:51:03
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/14/2021-03-14-estimate-pi-on-pi-day/wallhaven-yjk6ml.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/14/2021-03-14-estimate-pi-on-pi-day/wallhaven-yjk6ml.jpg?raw=true
summary: Happy π Day 2021! In this article I'll estimate the digits of π with random numbers and the probability of two integers being co-prime. What is the probability of two random integers being coprime? Euclidean Algorithm can be used to estimate π!
tags:
    - Python
    - Linear Algebra
    - Probability
categories: Mathematics
---
{% mathjax %}  {% endmathjax %}
# Introduction

Happy π Day 2021! In this article I'll estimate the digits of π with random numbers and the probability of two integers being co-prime. What is the probability of two random integers being coprime? Euclidean's Algorithm can be used to estimate π!

# Probability of Two Integers Being Coprime

Let {% mathjax %} q {% endmathjax %} denote the sought probability of two random integers being coprime. Pick an integer {% mathjax %} k {% endmathjax %} and two random numbers {% mathjax %} a {% endmathjax %} and {% mathjax %} b {% endmathjax %}. The probability that {% mathjax %} k {% endmathjax %} divides {% mathjax %} a {% endmathjax %} is {% mathjax %} \frac{1}{k} {% endmathjax %}, and the same holds for {% mathjax %} b {% endmathjax %}. Therefore, the probability that both {% mathjax %} a {% endmathjax %} and {% mathjax %} b {% endmathjax %} are divisible by {% mathjax %} k {% endmathjax %} equals {% mathjax %} \frac{1}{k^2} {% endmathjax %}. The probability that {% mathjax %} a {% endmathjax %} and {% mathjax %} b {% endmathjax %} have no other factors, i.e., that {% mathjax %} gcd(\frac{a}{k}, \frac{b}{k}) = 1 {% endmathjax %} equals {% mathjax %} q {% endmathjax %}, by our initial assumption. But {% mathjax %} gcd(\frac{a}{k}, \frac{b}{k}) = 1 {% endmathjax %} is equivalent to {% mathjax %} gcd(a, b) = k {% endmathjax %}. Assuming independence of the events, it follows that the probability that {% mathjax %} gcd(a, b) = k {% endmathjax %} equals {% mathjax %} \frac{1}{k^2} \cdot q = \frac{q}{k^2} {% endmathjax %}.

Now, {% mathjax %} k {% endmathjax %} was just one possibility for the greatest common divisor of two random numbers. Any number could be the {% mathjax %} gcd(a, b) {% endmathjax %}. Furthermore, since the events {% mathjax %} gcd(a, b) {% endmathjax %} are mutually exclusive (the {% mathjax %} gcd {% endmathjax %} of two numbers is unique) and the total probability of having a {% mathjax %} gcd {% endmathjax %} at all is 1 leads to:

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle 1 = \sum_{k=1}^{\infty} \frac{q}{k^2}
    {% endmathjax %}
</div>

implying that

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle q = [ \sum_{k=1}^{\infty} \frac{1}{k^2} ]^{-1} = \frac{6}{\pi^2}
    {% endmathjax %}
</div>

Or denoted in a statistical way

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    P(gcd(a, b)=1) = \frac{6}{\pi^2}
    {% endmathjax %}
</div>

## Euclidean Algorithm

In mathematics, the Euclidean Algorithm, or Euclid's Algorithm, is an efficient method for computing the greatest common divisor (GCD) of two integers (numbers), the largest number that divides them both without a remainder.

### Greatest Common Divisor

The greatest common divisor g is the largest natural number that divides both a and b without leaving a remainder.

```python
def gcd(a, b):
    while b != 0:
        t =  b
        b = a % b
        a = t
    return a
```

## Estimate Pi

Let's step back to the probability of two integers being coprime. We now know that

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    P(gcd(a, b)=1) = \frac{6}{\pi^2}
    {% endmathjax %}
</div>

it can be written as 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \frac{\# coprime}{\# total} = \frac{6}{\pi^2}
    {% endmathjax %}
</div>

so we get

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \pi = \sqrt{6 \times \frac{\# total}{\# coprime}}
    {% endmathjax %}
</div>

## Main Code

Talk is cheap. Show me the code.

```python
iterations = 400
num_total = 0
num_coprime = 0
pies = []

for idx in range(iterations):
    for i in range(500):
        a, b = random.randint(1, 1000000), random.randint(1, 1000000)
        if gcd(a, b) == 1:
            num_coprime += 1
        num_total += 1
    pi = math.sqrt(6 * num_total / num_coprime)
    pies.append(pi)
    pi_average = sum(pies)/len(pies)
    print(pi_average)
```

## Visualise

Let's visualise it dynamicly.

{% asset_img animation.gif %}

```python
iterations = 400
num_total = 0
num_coprime = 0
pies = []

plt.rcParams["figure.figsize"] = (15, 6)
ax = plt.axes()
plt.xlim(-1, iterations+1) 
plt.ylim(3, 3.3)
plt.axhline(y=3.1415, color='black', linestyle='-')
plt.grid()
for idx in range(iterations):
    for i in range(500):
        a, b = random.randint(1, 1000000), random.randint(1, 1000000)
        if gcd(a, b) == 1:
            num_coprime += 1
        num_total += 1
    pi = math.sqrt(6 * num_total / num_coprime)
    pies.append(pi)
    pi_average = sum(pies)/len(pies)
    ax.plot(
        [idx], [pi_average], 
        linestyle='-', lw=2, marker='o', color='tab:blue', markersize=2, alpha=0.5
    )
    plt.draw()
    custom_lines = [Line2D([0], [0], color='tab:blue', lw=1)]
    plt.legend(custom_lines, [f"pi={pi_average:.10f}"], loc="upper right")
    plt.pause(0.01)
plt.show()
```

# Conclusion

Today I show you how to estimate numbers π from random numbers using Euclidean algorithm. Happy coding! Cheers!

## References

1. https://www.cut-the-knot.org/m/Probability/TwoCoprime.shtml
2. https://en.wikipedia.org/wiki/Euclidean_algorithm
3. https://thecodingtrain.com/CodingChallenges/161-pi-from-random.html
4. https://www.youtube.com/watch?v=EvS_a921dBo
5. https://www.youtube.com/watch?v=RZBhSi_PwHU
6. https://www.youtube.com/watch?v=d-o3eB9sfls&list=PLCZeVeoafktVGu9rvM9PHrAdrsUURtLTo&index=54&ab_channel=3Blue1Brown