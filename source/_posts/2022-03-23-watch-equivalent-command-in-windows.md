---
title: WATCH Equivalent Command in Windows
top: false
cover: false
toc: true
mathjax: true
date: 2022-03-23 11:32:24
img: /images/wallhaven-wqve97.png
coverImg: /images/wallhaven-wqve97.png
summary: In Linux, the watch command is extremely handy for running a command on a regular basis and displaying the results. This is particularly useful if you need to keep track of any changes in the output of a command that is run repeatedly. The watch command has no direct equivalent in Windows, however the while loop in Windows PowerShell or the for loops in a Windows command-line prompt (CMD) can achieve the same result.
categories: Script
tags:
	- Linux
  - OS
  - CMD
---

# Introduction

In Linux, the `watch` command is extremely handy for running a command on a regular basis and displaying the results. This is particularly useful if you need to keep track of any changes in the output of a command that is run repeatedly. The watch command has no direct equivalent in Windows, however the while loop in Windows PowerShell or the for loops in a Windows command-line prompt (CMD) can achieve the same result.

# Using WATCH

Create a file called `watch.bat` inside `C:\Windows\System32\` folder.

```bash
@ECHO OFF
:loop
  cls
  %*
  timeout /t 5
goto loop
```

For example, prefixing the `docker ps` command with watch works like this:

```bash
watch docker ps
```

Or you can use a top-like utility for monitoring CUDA on a GPU by the following command:

```bash
watch nvidia-smi
```

# References

1. https://opensource.com/article/21/9/linux-watch-command
2. https://gist.github.com/gythialy/2800bcbec09df4664b3c
3. https://blog.miniasp.com/post/2011/08/30/Implement-watch-command-for-Windows