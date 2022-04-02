---
title: Story of Two Returns
top: false
cover: false
toc: true
mathjax: true
date: 2022-04-01 01:11:49
img: /images/wallhaven-72k2m3.jpg
coverImg: /images/wallhaven-72k2m3.jpg
summary: In finance, return is a profit on an investment. It can be used to gauge different metrics, all of which help determine how profitable a investment target is. A positive return represents a profit while a negative return marks a loss.
categories: Finance
tags:
	- Finance
	- Python
---

# Introduction

In finance, return is a profit on an investment. It can be used to gauge different metrics, all of which help determine how profitable a investment target is. A positive return represents a profit while a negative return marks a loss.

# Definition

Traditionally simple returns are denoted with a capital `R` (or {% mathjax %} R_{t}^{S} {% endmathjax %}) and log returns with a lower-case `r` (or {% mathjax %} R_{t}^{L} {% endmathjax %}).

<div style="display: flex;justify-content: center;">
	{% mathjax %} 
	\begin{aligned}
		R_{t} &= \frac{P_{t} - P_{t-1}}{P_{t-1}} = \frac{P_{t}}{P_{t-1}} - 1 \\
		r_{t} &= \ln(\frac{P_{t}}{P_{t-1}}) = \ln(P_{t}) - \ln(P_{t-1})
	\end{aligned}
	{% endmathjax %}
</div>

where {% mathjax %} P_{t} {% endmathjax %} is the price of the asset at time {% mathjax %} t {% endmathjax %}. We are defining the return from time {% mathjax %} t-1 {% endmathjax %} to time {% mathjax %} t {% endmathjax %}. The log function here is the natural logarithm. It is super easy to derive the relation between {% mathjax %} R_{t} {% endmathjax %} and {% mathjax %} r_{t} {% endmathjax %}:

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\begin{aligned}
	r_{t} &= \ln(R_{t} + 1) \\
	R_{t} &= e^{r_{t}} - 1
	\end{aligned}
	{% endmathjax %}
</div>

The first one is to go from simple to log returns, and the seconed one is to go from log return to simple return. Therefore, it can be shown that log returns are always smaller than simple returns.

It can also be deduced that using an approximation of the logarithm that {% mathjax %} \ln(1+x) \approx x {% endmathjax %}, if {% mathjax %} x {% endmathjax %} is near to zero. So to speak, if the simples return is near to zero, then it is in addition very comparable to the log return. If we want to illustrate {% mathjax %} \ln(1+x) = x {% endmathjax %} when {% mathjax %} x {% endmathjax %} is near to zero, we only need to prove that {% mathjax %} \lim_{x \to 0} \frac{\ln(1+)}{x} = 1 {% endmathjax %}.

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\begin{aligned}
		\lim_{x \to 0} \frac{\ln(1+x)}{x} &= \lim_{x \to 0} \frac{1}{x} \ln(1+x) \\
										 &= \lim_{x \to 0} \ln(1+x)^{\frac{1}{x}} \\
										 &= \ln e \\
										 &= 1
	\end{aligned}
	{% endmathjax %}
</div>

# Cumulative Returns

Let's take a look at a typical bank deposit to recap the concept of compounding. If you deposit $100 in a bank with a 10\% annual interest rate and a yearly compounding period. The following is an example of what you get in a year:

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\begin{aligned}
		D_{t} &= D_{0} (1 + \frac{R}{n})^{nt} \\
			  &= 100 (1 + \frac{0.1}{1})^{1 \times 1} \\
			  &= 110
	\end{aligned}
	{% endmathjax %}
</div>

What if we increase {% mathjax %} n {% endmathjax %} to 365 (i.e. daily compounding)?

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\begin{aligned}
		D_{t} &= D_{0} (1 + \frac{R}{n})^{nt} \\
			  &= 100 (1 + \frac{0.1}{365})^{365 \times 1} \\
			  &= 110.5155
	\end{aligned}
	{% endmathjax %}
</div>

In a daily compounding scenario, you earn interest on a daily basis, and those interest in turns makes you more interest, that is, interest on interest. How do all these relate to log returns? Let us now use market prices instead of bank deposits to illustrate the concept. The simple cumulative daily return is calculated by taking the cumulative product of the daily percentage change. This calculation is represented by the following equation:

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\frac{P_{n}}{P_{1}} - 1 = (1+R_{1}) \times (1+R_{2}) \times \ldots \times (1+R_{n}) - 1
	{% endmathjax %}
</div>

Compounding logarithmic returns over time can be calculated by the following:

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	\ln(\frac{P_{n}}{P_{1}}) = r_{1} + r_{2} + \ldots + r_{n}
	{% endmathjax %}
</div>

<figure>
  <img src="cumulative.png" width=100%>
</figure>

# Python Computation

In the Python language if you have a series of prices (in the code `price_df` is a `pd.DataFrame` object), you can compute the simple returns with:

```python
simple_returns = price_df['close'].pct_change()
simple_cumulative_returns = (1 + simple_returns).cumprod() - 1
```

As for log returns you can compute it with the following:

```python
log_returns = np.log(price_df['close']/price_df['close'].shift())
log_cumulative_returns = np.exp(log_returns.cumsum()) - 1
```

# Notes

It can be presented in the stock case that the logarithmic return has an advantage against the simple return since multi-period logarithmic return can be calculated as a sum of the one-period logarithmic returns. While the multi-period simple return is the product of the one-period simple returns, which can lead to computational problems for values close to zero. Additionally, we can see that if the simple return values are close to zero, the distribution of simple and logarithmic returns is extremely similar. This raises the question of whether the return type (simple or log) has an impact on the calculations and, as a result, on the outcomes.

# Conclusions

Our goal in this article was to explain the concepts of simple and logarithmic return, as well as the differences and connections between them. 

# References

1. https://www.portfolioprobe.com/2010/10/04/a-tale-of-two-returns/
2. https://core.ac.uk/download/pdf/161062652.pdf
3. https://investmentcache.com/magic-of-log-returns-concept-part-1/
4. https://ebrary.net/12923/management/logarithmic_returns