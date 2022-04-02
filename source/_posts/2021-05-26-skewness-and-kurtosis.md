---
title: Skewness and Kurtosis
top: false
cover: false
toc: true
mathjax: true
date: 2021-05-26 20:47:06
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/26/2021-05-26-skewness-and-kurtosis/wallhaven-k78j37.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/26/2021-05-26-skewness-and-kurtosis/wallhaven-k78j37.jpg?raw=true
summary: Statistics is a discipline of applied mathematics that deals with the gathering, describing, analysing, and inferring conclusions from numerical data. Differential and integral calculus, linear algebra, and probability theory are all used substantially in statistics' mathematical theories.
categories: Statistics
tags: 
	- Statistics
	- Skewness
	- Kurtosis
---

# Introduction

Statistics is a discipline of applied mathematics that deals with the gathering, describing, analysing, and inferring conclusions from numerical data. Differential and integral calculus, linear algebra, and probability theory are all used substantially in statistics' mathematical theories. Descriptive statistics, which explains the qualities of sample and population data, and inferential statistics, which uses those qualities to test hypotheses and make conclusions, are the two major disciplines of statistics. Skewness and Kurtosis are two common statistics tools for the study of manipulation of data.

# Skewness

Skewness is a distortion or asymmetry in a set of data that deviates from the symmetrical bell curve, or normal distribution. The curve is said to be skewed if it is displaced to the left or right. Skewness can be expressed as a measure of how far a given distribution deviates from a normal distribution. The skewness of a random variable {% mathjax %} X {% endmathjax %} is the third standardized moment {% mathjax %} \tilde \mu_{3} {% endmathjax %}, defined as:

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \tilde \mu_{3} = E \left[ \left( \frac{X - \mu}{\sigma} \right)^3 \right] = \frac{E \left[ \left( X - \mu \right)^3 \right]}{E \left[ \left( X - \mu \right)^2 \right]^{\frac{3}{2}}} = \frac{E[X^3] - 3 \mu \sigma^2 - \mu^3}{\sigma^3}
    {% endmathjax %}
</div>

Skewness is a descriptive statistic that can be used in conjunction with the histogram and the normal quantile plot to characterise the data or distribution. Many models assume normal distribution; i.e., data are symmetric about the mean. The normal distribution has a skewness of zero. But in reality, data points may not be perfectly symmetric. So, an understanding of the skewness of the dataset indicates whether deviations from the mean are going to be positive or negative.

- If skewness is less than -1 or greater than 1, the distribution is highly skewed.
- If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.
- If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.

The skewness for a normal distribution is zero, and any symmetric data should have a skewness near zero. Negative values for the skewness indicate data that are skewed left and positive values for the skewness indicate data that are skewed right. By skewed left, we mean that the left tail is long relative to the right tail. Similarly, skewed right means that the right tail is long relative to the left tail. If the data are multi-modal, then this may affect the sign of the skewness.

Take asset prices for example, beginning with the internet bubble of the late 1990s, deviations from "normal" returns have been more common in the recent two decades. In fact, asset returns are becoming increasingly skewed to the right. The terrorist acts of September 11, 2001, the housing bubble burst and following financial crisis, and the years of quantitative easing (QE) all contributed to this volatility.

# Kurtosis

Kurtosis, like skewness, is a statistical term used to quantify distribution. Unlike skewness, which distinguishes extreme values in one tail from those in the other, kurtosis assesses extreme values in both tails. 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    Kurt[X] = E \left[ \left( \frac{X - \mu}{\sigma} \right)^4 \right] = \frac{E \left[ \left( X - \mu \right)^4 \right]}{E \left[ \left( X - \mu \right)^2 \right]^{2}}
    {% endmathjax %}
</div>

This heaviness or lightness in the tails usually means that your data looks flatter (or less flat) compared to the normal distribution. The standard normal distribution has a kurtosis of 3, so if your values are close to that then your graph's tails are nearly normal.

Also an example for asset prices. For investors, a high kurtosis of the return distribution means that they will see more extreme returns (either positive or negative) than the normal + or - three standard deviations from the mean that the normal distribution of returns predicts. Kurtosis risk is the name given to this phenomenon.

# Conclusion

In this blog, we discuss the concept of skewness and kurtosis and their application in understanding the risk profiles of financial securities. In addition, we also glance over some common misconceptions regarding the calculation and interpretation of skewness and kurtosis.

# References

1. https://community.gooddata.com/metrics-and-maql-kb-articles-43/normality-testing-skewness-and-kurtosis-241
2. https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
3. https://codeburst.io/2-important-statistics-terms-you-need-to-know-in-data-science-skewness-and-kurtosis-388fef94eeaa