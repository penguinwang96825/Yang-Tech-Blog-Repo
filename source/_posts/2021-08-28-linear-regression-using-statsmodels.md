---
title: Linear Regression using statsmodels
top: false
cover: false
toc: true
mathjax: true
date: 2021-08-28 23:36:44
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/08/28/2021-08-28-linear-regression-using-statsmodels/wallhaven-8oky1j.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/08/28/2021-08-28-linear-regression-using-statsmodels/wallhaven-8oky1j.jpg?raw=true
summary: A linear regression to modelling the relationship between a scalar response and one or more explanatory variables is known as linear regression in statistics (also known as dependent and independent variables). Simple linear regression is used when there is only one explanatory variable; multiple linear regression is used when there are more than one.
categories: Statistics
tags:
	- Python
	- Statistics
---

# Introduction

Linear regression is a prediction model that assumes the dependent variable and the independent variable have a linear relationship. In this article, I'll fit a linear regression model on a synthetic dataset using the Python library "statsmodels."

# Linear Regression

A linear regression to modelling the relationship between a scalar response and one or more explanatory variables is known as linear regression in statistics (also known as dependent and independent variables). Simple linear regression is used when there is only one explanatory variable; multiple linear regression is used when there are more than one.

1. Simple Linear Regression: {% mathjax %} Y = \beta_0 + \beta X {% endmathjax %}
2. Multiple Linear Regression: {% mathjax %} Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots {% endmathjax %}

> **Are multiple and multivariate regression really different?**
> Multiple regression (aka multivariable regression) pertains to one dependent variable and multiple independent variables: {% mathjax %} y = f(x_1, x_2, \ldots, x_n) {% endmathjax %}. Multivariate regression pertains to multiple dependent variables and multiple independent variables: {% mathjax %} y_1, y_2, \ldots, y_m = f(x_1, x_2, \ldots, x_n) {% endmathjax %}.

## The OLS Assumptions

I divide OLS into five assumptions in this tutorial. Before you undertake regression analysis, you should be aware of all of them and take them into account.

* Linearity: {% mathjax %} Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots +\epsilon {% endmathjax %}
* No endogeneity: {% mathjax %} \sigma_{X, \epsilon} = 0: \forall x, \epsilon {% endmathjax %}
* Normality and homoscedasticity: {% mathjax %} \epsilon \sim N(0, \sigma^2) {% endmathjax %}
* No autocorrelation: {% mathjax %} \sigma_{\epsilon_i, \epsilon_j} = 0: \forall i \neq j {% endmathjax %}
* No multicollinearity: {% mathjax %} \rho_{x_i, x_j} \not\approx 1: \forall i, j; i \neq j {% endmathjax %}

These are the most important OLS assumptions for regression analysis.

## Implementation

Let's make a synthetic stock dataset for demonstration purposes.

<div style="display: flex; justify-content: center;">
    <table class="styled-table">
        <thead>
            <tr>
                <th>year</th>
                <th>month</th>
                <th>interest</th>
                <th>unemplyment</th>
                <th>price</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>2021</td>
                <td>12</td>
                <td>2.75</td>
                <td>5.3</td>
                <td>1464</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>11</td>
                <td>2.50</td>
                <td>5.3</td>
                <td>1394</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>10</td>
                <td>2.50</td>
                <td>5.3</td>
                <td>1357</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>9</td>
                <td>2.50</td>
                <td>5.3</td>
                <td>1293</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>8</td>
                <td>2.50</td>
                <td>5.4</td>
                <td>1256</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>7</td>
                <td>2.50</td>
                <td>5.6</td>
                <td>1254</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>6</td>
                <td>2.50</td>
                <td>5.5</td>
                <td>1234</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>5</td>
                <td>2.25</td>
                <td>5.5</td>
                <td>1195</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>4</td>
                <td>2.25</td>
                <td>5.5</td>
                <td>1159</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>3</td>
                <td>2.25</td>
                <td>5.6</td>
                <td>1167</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>2</td>
                <td>2.00</td>
                <td>5.7</td>
                <td>1130</td>
            </tr>
            <tr>
                <td>2021</td>
                <td>1</td>
                <td>2.00</td>
                <td>5.7</td>
                <td>1130</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>12</td>
                <td>2.00</td>
                <td>6.0</td>
                <td>1047</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>11</td>
                <td>1.75</td>
                <td>5.9</td>
                <td>965</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>10</td>
                <td>1.75</td>
                <td>5.8</td>
                <td>943</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>9</td>
                <td>1.75</td>
                <td>6.1</td>
                <td>958</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>8</td>
                <td>1.75</td>
                <td>6.2</td>
                <td>971</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>7</td>
                <td>1.75</td>
                <td>6.1</td>
                <td>949</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>6</td>
                <td>1.75</td>
                <td>6.1</td>
                <td>884</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>5</td>
                <td>1.75</td>
                <td>6.1</td>
                <td>866</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>4</td>
                <td>1.75</td>
                <td>5.9</td>
                <td>876</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>3</td>
                <td>1.75</td>
                <td>6.2</td>
                <td>822</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>2</td>
                <td>1.75</td>
                <td>6.2</td>
                <td>704</td>
            </tr>
            <tr>
                <td>2020</td>
                <td>1</td>
                <td>1.75</td>
                <td>6.1</td>
                <td>719</td>
            </tr>
        </tbody>
    </table>
</div>

The objective is to estimate the stock price using two independent variables: the interest rate and the unemployment rate. Multiple Linear Regression is seen in the Python code below.

```python
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler


def get_data():
	df = pd.read_csv("./stock.csv")
	features = df[["interest", "unemployment"]]
	labels = df["price"]
	return features, labels


if __name__ == "__main__":
	features, labels = get_data()
	scaler = MinMaxScaler()
	features = scaler.fit_transform(features)

	features = sm.add_constant(features)
	model = sm.OLS(labels, features).fit()
	predictions = model.predict(features)
	print(model.summary())
```

When you run the Python code, you'll get the following result.

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                  price   R-squared:                       0.898
Model:                            OLS   Adj. R-squared:                  0.888
Method:                 Least Squares   F-statistic:                     92.07
Date:                Sat, 28 Aug 2021   Prob (F-statistic):           4.04e-11
Time:                        23:52:11   Log-Likelihood:                -134.61
No. Observations:                  24   AIC:                             275.2
Df Residuals:                      21   BIC:                             278.8
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       1077.3223     91.490     11.775      0.000     887.059    1267.586
x1           345.5401    111.367      3.103      0.005     113.940     577.140
x2          -225.1319    106.155     -2.121      0.046    -445.893      -4.371
==============================================================================
Omnibus:                        2.691   Durbin-Watson:                   0.530
Prob(Omnibus):                  0.260   Jarque-Bera (JB):                1.551
Skew:                          -0.612   Prob(JB):                        0.461
Kurtosis:                       3.226   Cond. No.                         14.4
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

## Interpretation

* **F-statistic** in linear regression is comparing your produced linear model for your variables against a model that replaces your variables' effect to 0, to find out if your group of variables are statistically significant.
* **Prob (F-Statistic)** uses this number to tell you the accuracy of the null hypothesis, or whether it is accurate that your variables' effect is 0.
* **AIC** and **BIC** are both used to compare the efficacy of models in the process of linear regression, using a penalty system for measuring multiple variables. These numbers are used for feature selection of variables.
* **R-squared** is the measurement of how much of the independent variable is explained by changes in our dependent variables.
* **Adjusted. R-squared** measures how well the model fits the data. R-squared values vary from 0 to 1, with a greater value indicating a better match if specific criteria are met. The adjusted R-squared penalizes the R-squared formula based on the number of variables, therefore a lower adjusted score may be telling you some variables are not contributing to your model’s R-squared properly.
* **interest coefficient** represents the change in the output `price` due to a change of one unit in the interest rate (everything else held constant).
* **unemployment coefficient** represents the change in the output `price` due to a change of one unit in the unemployment rate (everything else held constant).
* **std err** reflects the level of accuracy of the coefficients. The lower the number, the higher the level of accuracy.
* **P >|t|** is the p-value. Statistical significance is defined as a p-value of less than 0.05.
* **Confidence Interval** denotes the range of possibilities for our coefficients (with a likelihood of 95 percent).
* **Omnibus** describes the normalcy of the distribution of our residuals using skew and kurtosis as measurements. A 0 would indicate perfect normalcy.
* **Prob(Omnibus)** is a statistical test measuring the probability the residuals are normally distributed. A 1 would indicate perfectly normal distribution.
* **Skew** is a measurement of symmetry in our data, with 0 being perfect symmetry.
* **Kurtosis** measures the peakiness of our data, or its concentration around 0 in a normal curve. Higher kurtosis implies fewer outliers.
* **Durbin-Watson** falls between 0 and 4. To be more into detail, 2 denotes no autocorrelation, and if the figure is lower than 1 or higher than 3 cause an alarm.
* **Jarque-Bera (JB)** and **Prob(JB)** are alternate methods of measuring the same value as `Omnibus` and `Prob(Omnibus)` using skewness and kurtosis.

Recall that the equation for the MLR is {% mathjax %} Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots {% endmathjax %}. So for our example, it would look like this: 

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    price = 1077.3223 + 345.5401 \times interest - 225.1319 \times unemployment
    {% endmathjax %}
</div>

# Conclusions

Most linear and multiple linear regression models are based on OLS. I hope this post clarified some topics for you, and I look forward to hearing from you in the comments section. Happy statistics!

# References

1. https://365datascience.com/tutorials/statistics-tutorials/ols-assumptions/
2. https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a