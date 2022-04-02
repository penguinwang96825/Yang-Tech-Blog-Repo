---
title: One-way ANOVA Implemented from Scratch
top: false
cover: false
toc: true
mathjax: true
date: 2021-08-29 01:36:50
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/08/29/2021-08-29-anova-using-statsmodels/wallhaven-pky9yp.png?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/08/29/2021-08-29-anova-using-statsmodels/wallhaven-pky9yp.png?raw=true
summary: ANOVA (ANalysis Of VAriance) test used to compare the means of more than 2 groups, whilst t-test can only be used to compare 2 groups. ANOVA uses variance-based F test to check the group mean equality. There are two main types of ANOVA, one-way (one factor) and two-way (two factors) ANOVA.
categories: Statistics
tags:
	- Python
	- Statistics
---

# Introduction

ANOVA (ANalysis Of VAriance) test used to compare the means of more than 2 groups, whilst t-test can only be used to compare 2 groups. ANOVA uses variance-based F test to check the group mean equality. There are two main types of ANOVA, one-way (one factor) and two-way (two factors) ANOVA.

# Hypothesis

* Null hypothesis ({% mathjax %} H_0 {% endmathjax %}): Groups means are equal ({% mathjax %} \mu_1 = \mu_2 = \ldots = \mu_p {% endmathjax %})
* Alternative hypothesis ({% mathjax %} H_1 {% endmathjax %}): At least, one group mean is different from other groups.

# Implementation

In our example, there are four treatments, which are A, B, C, and D. Treatments are independent variable and termed as factor. As there are four types of treatments, treatment factor has four levels.

<img src="treatments.png" width=400>

First, we need to calculate `sum of squares within groups`. Second, calculate `sum of squares between groups`. Third, calculate `total sum of squares`. Finally, compute `F ratio` and see if it falls in rejection region or not. When we carry out an ANOVA on a computer, we will get an ANOVA table, as shown below:

| Source of Variation | Sums of Squares (SS) | Degrees of Freedom (df) | Mean Squares (MS) | F score |
| :----: | :----: | :----: | :----: | :----: |
| Between Treatments | SSB | k-1 | MSB | F |
| Within Treatments  | SSW | N-k | MSW | - |
| Total              | SST | N-1 | -   | - |

where k is the number of treatments or independent comparison groups, and N is the total number of observations or total sample size. In addition, {% mathjax %} SST = SSB + SSW {% endmathjax %}, thus if two sums of squares are known, the third can be computed from the other two.

## Sum of Squares between Groups (SSB)

Sum of Squares Between Groups is the variability due to interaction between the groups. Sometimes known as the Sum of Squares of the Model.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle SSB = n \sum_{i=1}^{k} (\bar{y_i} - \bar{y})^2
    {% endmathjax %}
</div>

<img src="ssb.png" width=400>

## Sum of Squares within Groups (SSW)

Variability in the data as a result of group differences. The formula below can be used to calculate the Sum of Squares Within Groups.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle SSW = \sum_{i=1}^{k} \sum_{j=1}^{n} (y_{ij} - \bar{y_i})^2
    {% endmathjax %}
</div>

<img src="sse.png" width=400>

## Sum of Squares Total (SST)

Sum of Squares Total will be needed to calculate eta-squared ({% mathjax %} \eta^2 {% endmathjax %}) later. This is the total variability in the data.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \displaystyle SST = \sum_{i=1}^{k} \sum_{j=1}^{n} (y_{ij} - \bar{y})^2
    {% endmathjax %}
</div>

In addition, we'll figure out the effect size. We'll start with the widely known eta-squared formula.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \eta^2 = \frac{SSB}{SST}
    {% endmathjax %}
</div>

However, because it is based solely on sums of squares from the sample, eta-squared is biased. There is no adjustment for the fact that our goal is to assess the effect size in the population. As a result, we can utilise the less biassed omega-squared effect size estimate.

<div style="display: flex;justify-content: center;">
    {% mathjax %}
    \omega^2 = \frac{SSB - (DF_{between} \times MSW)}{SST + MSW}
    {% endmathjax %}
</div>

In summary, to test {% mathjax %} H_0 : \mu_1 = \mu_2 = \ldots = \mu_k = \mu {% endmathjax %} use the statistic {% mathjax %} F = \frac{MSB}{MSW} {% endmathjax %} and compare this to the F distribution with {% mathjax %} k-1 {% endmathjax %} and {% mathjax %} N-k {% endmathjax %} degrees of freedom.

## Python Code

For the aim of constructing an ANOVA table, there are three primary functions.

* `get_ssb_and_msb()`: get the SSB and MSB.
* `get_ssw_and_msw()`: get the SSW and MSW.
* `get_f()`: get the F score and p-value.

```python
import pandas as pd
from scipy import stats
from tabulate import tabulate


def get_data():
    df = pd.read_csv("treatments.csv")
    return df

def get_ssb_and_msb(df, group_col, value_col):
    k = len(pd.unique(df[group_col]))
    N = len(df.values)
    n = df.groupby(group_col).size()[0]
    df_between = k - 1
    df_within = N - k
    df_total = N - 1
    mean_of_group = df[value_col].sum() / N
    mean_of_each_group = df.groupby(group_col).agg({value_col:"mean"})[value_col]
    SSB = n * sum([(m-mean_of_group)**2 for m in mean_of_each_group])
    MSB = SSB / df_between
    return SSB, MSB

def get_ssw_and_msw(df, group_col, value_col):
    k = len(pd.unique(df[group_col]))
    N = len(df.values)
    n = df.groupby(group_col).size()[0]
    df_between = k - 1
    df_within = N - k
    df_total = N - 1
    SS = df[value_col] - df.groupby(group_col, axis=0).transform('mean')[value_col]
    SSW = sum([v**2 for v in SS.values])
    MSW = SSW / df_within
    return SSW, MSW

def get_f(df, group_col, value_col):
    k = len(pd.unique(df[group_col]))
    N = len(df.values)
    n = df.groupby(group_col).size()[0]
    df_between = k - 1
    df_within = N - k
    df_total = N - 1
    SSB, MSB = get_ssb_and_msb(df, group_col, value_col)
    SSW, MSW = get_ssw_and_msw(df, group_col, value_col)
    F = MSB / MSW
    P = stats.f.sf(F, df_between, df_within)
    return F, P

def get_anova_table(df, group_col, value_col):
    k = len(pd.unique(df[group_col]))
    N = len(df.values)
    n = df.groupby(group_col).size()[0]
    df_between = k - 1
    df_within = N - k
    df_total = N - 1

    SSB, MSB = get_ssb_and_msb(df, group_col, value_col)
    SSW, MSW = get_ssw_and_msw(df, group_col, value_col)
    F, P = get_f(df, group_col, value_col)

    if P <= 0.001:
        significance_stars = "(***)"
    elif P > 0.001 and P <= 0.01:
        significance_stars = "(**)"
    elif P > 0.01 and P <= 0.05:
        significance_stars = "(*)"
    elif P > 0.05 and P <= 0.1:
        significance_stars = "(+)"
    elif P > 0.1 and P <= 1.0:
        significance_stars = ""

    table = pd.DataFrame({
        "Source of Variation": ["Between Treatments", "Within Treatments", "Total"], 
        "Sum of Squares (SS)": [SSB, SSW, SSB+SSW], 
        "Degrees of Freedom (df)": [df_between, df_within, df_total], 
        "Mean Squares (MS)": [MSB, MSW, "-"], 
        "F Score": [F, "-", "-"], 
        "PR(>F)": [f"{P}{significance_stars}", "-", "-"]
    })

    return table

if __name__ == "__main__":
    group_col = "treatments"
    value_col = "value"

    SSB, MSB = get_ssb_and_msb(df, group_col, value_col)
    SSW, MSW = get_ssw_and_msw(df, group_col, value_col)
    F, P = get_f(df, group_col, value_col)

    table = get_anova_table(df, group_col, value_col)
    print(tabulate(table, headers='keys', tablefmt='psql'))
```

With the help of `statsmodels` library, we can get the same result as we implemented from scratch.

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

if __name__ == "__main__":
	group_col = "treatments"
    value_col = "value"

    df = get_data()
    model = ols(f'{value_col} ~ C({group_col})', data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    print(table)
```

## Analysis

### Build ANOVA Table

Draw a boxplot for each numeric variable in a DataFrame.

<img src="boxplot.png" width=600>

In the final part of this section, I am going to carry out the ANOVA table using the implemented python code from scratch.

<img src="table.png" width=800>

The p-value obtained from ANOVA analysis is significant (smaller than 0.05), and therefore, it can be concluded that there are significant differences among treatments. I have wrote this {% post_link 2021-03-12-easiest-explanation-for-p-value article %} explaining what p-value is. We know that treatment differences are statistically significant thanks to ANOVA analysis, but ANOVA doesn't tell us which treatments are significantly different. To deal with this circumstance, a multiple pairwise comparison analysis employing Tukey's honestly significantly differenced  (HSD) test can be used.

### Test ANOVA Assumptions

Perform visualisation technique on this data first.

```python
from bioinfokit.analys import stat

res = stat()
res.anova_stat(df=df, res_var=value_col, anova_model=f'{value_col} ~ C({group_col})')

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
ax = plt.gca()
sm.qqplot(res.anova_std_residuals, line='45', ax=ax)
plt.title("QQ-plot from standardized residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.grid()
plt.subplot(1, 2, 2)
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
plt.title("Residuals histogram")
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.grid(axis="y")
plt.tight_layout()
plt.show()
```

<img src="vis.png" width=800>

As shown in the figure on the left, because the standardised residuals are centred around the 45-degree line, the residuals appear to be somewhat normally distributed. The distribution appears to be fairly normal in the histogram, implying that residuals are distributed normally, as shown in the figure on the right. Shapiro-Wilk test can be used to check the normal distribution of residuals. Null hypothesis is that data is drawn from normal distribution.

# Conclusions

ANOVA has allowed us to assess statistically if sample differences may be extrapolated to population differences in this article. I hope you found this article helpful, and I wish you luck with your one-way ANOVA. Thank you for taking the time to read this!

# References

1. https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
2. https://www.youtube.com/watch?v=-yQb_ZJnFXw&t=56s
3. https://www.reneshbedre.com/blog/anova.html