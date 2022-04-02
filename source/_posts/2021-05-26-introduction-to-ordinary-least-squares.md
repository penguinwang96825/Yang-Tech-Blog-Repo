---
title: Introduction to Ordinary Least Squares
top: false
cover: false
toc: true
mathjax: true
date: 2021-05-26 21:48:39
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/26/2021-05-26-introduction-to-ordinary-least-squares/wallhaven-e78jpo.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/26/2021-05-26-introduction-to-ordinary-least-squares/wallhaven-e78jpo.jpg?raw=true
summary: In empirical finance and many other domains, linear regression and the closely related linear prediction theory are commonly used statistical methods. Because of the wide range of applications, basic linear regression courses normally concentrate on the mathematically simplest scenario, which can be used in a variety of other applications.
categories: Statistics
tags:
	- OLS
	- Linear Regression
---

# Introduction

In empirical finance and many other domains, linear regression and the closely related linear prediction theory are commonly used statistical methods. Because of the wide range of applications, basic linear regression courses normally concentrate on the mathematically simplest scenario, which can be used in a variety of other applications.

# Ordinary least squares (OLS)

A linear regression model relates the output (or response) {% mathjax %} y_t {% endmathjax %} to {% mathjax %} q {% endmathjax %} input (or predictor) variables {% mathjax %} x_{t1}, \ldots, x_{tq} {% endmathjax %}, which are also called regressors, via

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    y_t = a + b_1 x_{t1} + \ldots + b_q x_{tq} + \epsilon_{t}
    {% endmathjax %}
</div>

where the {% mathjax %} \epsilon_{t} {% endmathjax %} are unobservable random errors that are assumed to have zero means. The coefficients are unknown parameters that have to be estimated from the observed input-output vectors.

To fit a regression model to the observed data, the method of least squares chooses {% mathjax %} a, b_1, \ldots, b_p {% endmathjax %} to minimise the residual sum of squares (RSS).

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    RSS = \sum_{t=1}^{n} \left[ \left( a + b_1 x_{t1} + \ldots + b_q x_{tq} \right) \right]^2
    {% endmathjax %}
</div>

Setting to 0 the partial derivative of RSS with respect to {% mathjax %} a, b_1, \ldots, b_p {% endmathjax %} yields {% mathjax %} q+1 {% endmathjax %} linear equations, whose solution gives the OLS estimates. The regression model can be written in matrix form as

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    Y = X \beta + \epsilon
    {% endmathjax %}
</div>

The vector of least squares estimates of the {% mathjax %} \beta_{i} {% endmathjax %} is given by 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \hat{\beta} = (X^T X)^{-1} X^T Y
    {% endmathjax %}
</div>

Using this matrix notation, RSS can be written as

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    RSS = (Y - X \hat{\beta})^T (Y - X \hat{\beta})
    {% endmathjax %}
</div>

# Statistical Properties of OLS Estimates

1. {% mathjax %} x_{tj} {% endmathjax %} are nonrandom constants and {% mathjax %} X {% endmathjax %} has full rank p, where {% mathjax %} p = q + 1 {% endmathjax %}.
2. {% mathjax %} \epsilon_{t} {% endmathjax %} are unobserved random disturbances with {% mathjax %} E[\epsilon_{t}] = 0 {% endmathjax %}
3. {% mathjax %} Var(\epsilon_{t}) = \sigma^{2} {% endmathjax %} and {% mathjax %} Cov(\epsilon_{i}, \epsilon_{j}) = 0 {% endmathjax %} for {% mathjax %} i \neq j {% endmathjax %}
4. {% mathjax %} \epsilon_{t} {% endmathjax %} are independent {% mathjax %} N(0, \sigma^{2}) {% endmathjax %}, where {% mathjax %} N(0, \sigma^{2}) {% endmathjax %} denotes the normal distribution with mean {% mathjax %} \mu {% endmathjax %} and variance {% mathjax %} \sigma^{2} {% endmathjax %}.

# Case Study

We illustrate the application of this methods in a case study that relates the daily log returns of the stock of Microsoft Corporation to those of several computer and software companies.

{% asset_img returns.png %}

Starting with the full model, we find that the stocks hp and sunw, with relatively small partial F-statistics, are not significant at the 5% significance level. If we set the cutoff value at {% mathjax %} F^{\star} = 10 {% endmathjax %}, which corresponds to a significance level smaller than 0.01, then hp and sunw are removed from the set of predictors after the first step. We then refit the model with the remaining predictors and repeat the backward elimination procedure with the cutoff value {% mathjax %} F^{\star} = 10 {% endmathjax %} for the partial F-statistics. Proceeding stepwise in this way, the procedure terminates with six predictor variables: aapl, adbe, dell, gtw, ibm, orcl. The regression model can be illustrated as

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    msft = \beta_0 + \beta_1 aapl + \beta_2 adbe + \beta_3 dell + \beta_4 gtw + \beta_5 ibm \beta_6 orcl + \epsilon
    {% endmathjax %}
</div>

## Regression coefficients of the full model.

{% asset_img coefficients.png %}

## Regression coefficients of the selected regression model.

{% asset_img selected.png %}

The selected model shows that, in the collection of stocks we studied, the msft daily log return is strongly influenced by those of its competitors.

# Conclusion

The importance of regression analysis lies in the fact that it provides a powerful statistical method that allows a business to examine the relationship between two or more variables of interest.

# References

1. Lai and Xing, Statistical Models and Methods for Financial Markets (2008)