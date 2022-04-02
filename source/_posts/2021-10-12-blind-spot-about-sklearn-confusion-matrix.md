---
title: Blind Spot about Sklearn Confusion Matrix
top: false
cover: false
toc: true
mathjax: true
date: 2021-10-12 17:54:33
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/10/12/2021-10-12-blind-spot-about-sklearn-confusion-matrix/wallhaven-72rd8e.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/10/12/2021-10-12-blind-spot-about-sklearn-confusion-matrix/wallhaven-72rd8e.jpg?raw=true
summary: Evaluate the model we developed while performing research for either machine learning or deep learning projects is crucial. The best technique to see if the predicted value is well-classified is to use a confusion matrix. The confusion matrix function in the sklearn package, however, has a different interpretation than the one we usually find on other websites.
categories: Data Science
tags:
	- Python
	- Sklearn
	- Data Science
---

# Introduction

Evaluate the model we developed while performing research for either machine learning or deep learning projects is crucial. The best technique to see if the predicted value is well-classified is to use a confusion matrix. The confusion matrix function in the sklearn package, however, has a different interpretation than the one we usually find on other websites.

# Compare WIKI with SKLEARN

<figure>
  <img src="cm.png" width=400>
</figure>

In wiki page, we can see that each row of the matrix represents the instances in an actual class (ground truth) while each column represents the instances in a predicted class. But in sklearn `confusion_matrix()` function, each row of the matrix represents the instances in an predicted class while each column represents the instances in a actual class.

```python
from sklearn import metrics

y_true = ["cat", "dog", "cat", "cat", "dog", "penguin"]
y_pred = ["dog", "dog", "cat", "cat", "dog", "cat"]
metrics.confusion_matrix(y_true, y_pred, labels=["cat", "dog", "penguin"])
```

This will return

```python
array([[2, 1, 0],
       [0, 2, 0],
       [1, 0, 0]], dtype=int64)
```

# Conclusion

A confusion matrix is a two-row, two-column table in predictive analytics that provides the value of false positives, false negatives, true positives, and true negatives. This enables for more in-depth analysis than simply the fraction of right classifications, such as accuracy, f1, precision, and recall scores.

# References

1. https://blog.csdn.net/m0_38061927/article/details/77198990
2. https://en.wikipedia.org/wiki/Confusion_matrix