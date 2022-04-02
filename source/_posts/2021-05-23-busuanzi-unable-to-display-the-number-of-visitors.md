---
title: Busuanzi Unable to Display the Number of Visitors
top: false
cover: false
toc: true
mathjax: true
date: 2021-05-23 18:56:47
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/23/2021-05-23-busuanzi-unable-to-display-the-number-of-visitors/wallhaven-yj8k1g.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/23/2021-05-23-busuanzi-unable-to-display-the-number-of-visitors/wallhaven-yj8k1g.jpg?raw=true
summary: When adding statistics to the number of people and visits to the Hexo blog, according to the standard process of writing, pushing the number of people and visits after running found that the number of people simply do not load. Looking through the information found that the reason, busuanzi because in 2018/10/12 its domain name expired, so the number of people can not be displayed. 
categories: Script
tags:
	- Busuanzi
	- JavaScript
---

# Introduction

When adding statistics to the number of people and visits to the blog, according to the standard process of writing, pushing the number of people and visits after running found that the number of people simply do not load. Looking through the information found that the reason, busuanzi because in 2018/10/12 its domain name expired, so the number of people can not be displayed. 

# Solution

If your theme is an old version, please update the file `busuanzi.pure.mini.js` in `source\lib\others\` folder. Copy the code from [here](http://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js), and paste it in the file `busuanzi.pure.mini.js`. After modification, you can successfully see the number of visitors.

{% asset_img visitor.png %}

# References

1. https://busuanzi.ibruce.info/