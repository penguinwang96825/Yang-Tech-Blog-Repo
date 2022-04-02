---
title: Notes on Feature Engineering
top: false
cover: false
toc: true
mathjax: true
date: 2021-03-22 18:57:09
img: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/22/2021-03-22-notes-on-basic-feature-engineering/wallhaven-4gj65l.jpg?raw=true
coverImg: https://github.com/penguinwang96825/Hexo-Blog/blob/master/2021/03/22/2021-03-22-notes-on-basic-feature-engineering/wallhaven-4gj65l.jpg?raw=true
summary: Without sufficient data and suitable features, the most powerful model structure cannot get satisfactory output. As a classic saying goes, "Garbage in, garbage out." For a machine learning problem, the data and features often determine the upper limit of the results, while the selection of models, algorithms and optimization are gradually approaching this upper limit.
tags:
	- Data Science
	- Statistics
categories: Data Science
---

Without sufficient data and suitable features, the most powerful model structure cannot get satisfactory output. As a classic saying goes, "Garbage in, garbage out." For a machine learning problem, the data and features often determine the upper limit of the results, while the selection of models, algorithms and optimization are gradually approaching this upper limit.

# Introduction

Feature engineering, as the name implies, is a series of engineering processes on raw data to distill it into features that can be used as input for algorithms and models. In essence, feature engineering is a process of representing the data.

Typically, there are two types of data:

1. Structured Data: The structured data type can be viewed as a table in a relational database, with each column is clearly defined and contains two basic types: numeric and categorical; each row of data represents a sample. Each row of data represents a sample of information.
2. Unstructured Data: Unstructured data mainly includes text, image, audio and video data. The information it contains cannot be represented by a simple numerical value, and there is no clear definition of categories, and each piece of data. The size of each piece of data varies.

# Feature Normalisation

Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between an interval. Feature normalisation will make different indicators comparable. For example, to analyze the impact of a person's height and weight on health, if using meters (m) and kilograms (kg) as units, then the height characteristics would be in the range of values from 1.6 to 1.8m, and weight characteristics would be in the range of 50 to 100 kg, and the results of the analysis would clearly favor the weight characteristics with a greater numerical difference.

The two main methods most commonly used are as follows.

1. Min-Max Scaling

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
	{% endmathjax %}
</div>

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
```

2. Z-Score Normalisation

<div style="display: flex;justify-content: center;">
	{% mathjax %}
	Z = \frac{x-\mu}{\sigma}
	{% endmathjax %}
</div>

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```

Normalisation technique heavily depends on an algorithm being used. Indeed many estimators are designed with the assumption that each feature takes values close to zero or more importantly that all features vary on comparable scales. In particular, metric-based and gradient-based estimators (linear regression, logistic regression, support vector machine, neural network) often assume approximately standardized data (centered features with unit variances). A notable exception are decision tree-based estimators (random forest, gradient boosting tree) that are robust to arbitrary scaling of the data.

# Categorical Feature

Mainly, categorical feature indicates the gender (male or female), blood type (A, B,
AB, O), and other characteristics that only take values within a limited number of options. The most commonly used encoding methods are: ordinal encoding, one-hot encoding, binary encoding, etc.

## Ordinal Encoding

Ordinal encoding is usually used to handle data with size relationships between categories. For example, grades can be divided into low, medium and high, and there is a "high > medium > low" ranking relationship. Ordinal encoding will assign a numerical ID to category-type features with a numeric ID, e.g., 3 for high, 2 for medium, and 1 for low, which still retains the size relationship after conversion. 

## One-hot Encoding

One-hot encoding is usually used to deal with features that do not have size relationships between categories. In the "color" variable example, there are 3 categories and therefore 3 binary variables are needed. A "1" value is placed in the binary variable for the color and "0" values for the other colors. Such as (1, 0, 0) for red, (0, 1, 0) fo green, and (0, 0, 1) for blue.

## Binary Encoding

Binary encoding is divided into two main steps, first assigning a category ID to each category using the ordinal encoding, and then the binary code corresponding to the category ID is used as the result. Take color for instance, the ID for red is 1, so the binary representation is 001; the ID for green is 2, so the binary representation is 010; the ID for blue is 3, so the binary representation is 011. It can be seen that binary encoding is essentially a hash mapping of IDs using binary representation, which eventually yields a 0/1 feature vector with fewer dimensions than the one-hot encoding, it will save storage space.