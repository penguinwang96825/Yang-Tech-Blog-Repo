---
title: Basic Investment Models and Their Statistical Analysis
top: false
cover: false
toc: true
mathjax: true
date: 2021-05-26 22:54:04
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/26/2021-05-26-basic-investment-models-and-their-statistical-analysis/wallhaven-y8qlmk.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/26/2021-05-26-basic-investment-models-and-their-statistical-analysis/wallhaven-y8qlmk.jpg?raw=true
summary: Three cornerstones of quantitative finance are asset returns, interest rates, and volatilities. They appear in many fundamental formulas in finance. In this article, we consider their interplay and the underlying statistical issues in a classical topic in quantitative finance.
categories: Statistics
tags:
	- Statistics
	- Asset
---

# Introduction

Three cornerstones of quantitative finance are asset returns, interest rates, and volatilities. They appear in many fundamental formulas in finance. In this article, we consider their interplay and the underlying statistical issues in a classical topic in quantitative finance.

# Asset Returns

## One-period Net Returns and Gross Returns

Let {% mathjax %} P_{t} {% endmathjax %} denote the asset price at time {% mathjax %} t {% endmathjax %}. Suppose the asset does not have dividends over the period from time {% mathjax %} t-1 {% endmathjax %} to time {% mathjax %} t {% endmathjax %}. Then the one-period net return on this asset is

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    R_{t} = \frac{P_{t} - P_{t-1}}{P_{t-1}}
    {% endmathjax %}
</div>

which is the profit rate of holding the asset during the period. Another concept is the gross return {% mathjax %} \frac{P_{t}}{P_{t-1}} {% endmathjax %}, which is equal to {% mathjax %} 1 + R_t {% endmathjax %}.

## Multiperiod Returns

One-period returns can be extended to the multiperiod case as follows. The gross return over {% mathjax %} k {% endmathjax %} periods is then defined as

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle 1 + R_{t}(k) = \frac{P_{t}}{P_{t-1}} = \prod_{j=0}^{k-1} (1 + R_{t-j})
    {% endmathjax %}
</div>

and the net return over these periods is {% mathjax %} R_{t}(k) {% endmathjax %}. In practice, we usually use years as the time unit. The annualized gross return for holding an asset over {% mathjax %} k {% endmathjax %} years is {% mathjax %} (1 + R_{t}(k))^{\frac{1}{k}} {% endmathjax %} and the annualized net return is {% mathjax %} (1 + R_{t}(k))^{\frac{1}{k}} - 1 {% endmathjax %}.

## Continuously Compounded Return (Log Return)

Let {% mathjax %} p_t = log(P_{t}) {% endmathjax %}. The logarithmic return or continuously compounded return on an asset is defined as

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    r_t = log(\frac{P_{t}}{P_{t-1}}) = p_t - p_{t-1}
    {% endmathjax %}
</div>

One property of log returns is that, as the time step {% mathjax %} \delta t {% endmathjax %} of a period approaches 0, the log return {% mathjax %} r_t {% endmathjax %} is approximately equal to the net return:

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    r_t = log(\frac{P_{t}}{P_{t-1}}) = log(1 + R_t) \approx R_t
    {% endmathjax %}
</div>

# Asset Prices and Returns

The mean and standard deviation (SD, also called volatility) of the annual log return {% mathjax %} r_{year} {% endmathjax %} are related to those of the monthly log return {% mathjax %} r_{month} {% endmathjax %} by

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    E(r_{year}) = 12 E(r_{month})
    {% endmathjax %}
</div>

and

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    SD(r_{year}) = \sqrt{12} SD(r_{month})
    {% endmathjax %}
</div>

For daily returns, we consider only the number of trading days in the year (often taken to be 252). The convention above is for relating the annual mean return and its volatility to their monthly or daily counterparts. This convention is based on the `i.i.d.` assumption of daily returns.

# Conclusion

Market data are actually much more complicated and voluminous than those summarized in the financial press. Transaction databases consist of historical prices, traded quantities, and bidask prices and sizes, transaction by transaction. These 'high-frequency' data provide information on the 'market microstructure.'