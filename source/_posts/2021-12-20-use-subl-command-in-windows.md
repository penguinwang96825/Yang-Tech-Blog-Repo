---
title: Use SUBL Command in Windows
top: false
cover: false
toc: true
mathjax: true
date: 2021-12-20 23:25:47
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/12/20/2021-12-20-use-subl-command-in-windows/wallhaven-wq37p6.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/12/20/2021-12-20-use-subl-command-in-windows/wallhaven-wq37p6.jpg?raw=true
summary: It's sometimes easier to edit a file from terminal by using Sublime Text, and yet it is not the default in Windows. This tutorial will show you how to use the command line to open files with Sublime Text in Windows.
categories: Script
tags:
	- Windows
	- CMD
---

# Introduction

It's sometimes easier to edit a file from terminal by using Sublime Text, and yet it is not the default in Windows. This tutorial will show you how to use the command line to open files with Sublime Text in Windows.

# Steps

1. Download the latest Sublime Text from [here](https://www.sublimetext.com/3).

2. Append `subl.exe` to the system `PATH` environment. In my case, the path is at `C:\Program Files\Sublime Text 3`. Therefore, open the terminal and type in `set PATH=%PATH%;"C:\Program Files\Sublime Text 3"`. `set` is a command that changes cmd's environment variables only for the current cmd session. The `%PATH%` part expands to the current value of `PATH`, and the path afterwards is then concatenated to it.

3. Use in your terminal/console `subl` as a command to open whatever file. For example to open `README.md`, just type in `subl README.md`.

# Conclusions

Sublime Text is a powerful text editor that is popular among programmers. To boost their productivity, all programmers should be aware of these tips.