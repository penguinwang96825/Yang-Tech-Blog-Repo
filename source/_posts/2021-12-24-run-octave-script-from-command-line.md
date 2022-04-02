---
title: Run Octave Script from Command Line
top: false
cover: false
toc: true
mathjax: true
date: 2021-12-24 19:18:41
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/12/24/2021-12-24-run-octave-script-from-command-line/wallhaven-1ko679.png?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/12/24/2021-12-24-run-octave-script-from-command-line/wallhaven-1ko679.png?raw=true
summary: Octave is an open-source replacement for MATLAB, a software and programming environment for numerical arithmetic and data analysis. Sometimes it's more convenient to run Octave scripts from the command line.
categories: Script
tags:
	- MATLAB
	- Octave
	- CMD
---

# Introduction

Octave is an open-source replacement for MATLAB, a software and programming environment for numerical arithmetic and data analysis. Sometimes it's more convenient to run Octave scripts from the command line.

# Steps

1. Install Octave from [here](https://www.gnu.org/software/octave/download).

2. Add `octave-cli.exe` to the system `PATH` environment. In my case, the path is `C:\Program Files\GNU Octave\Octave-6.4.0\mingw64\bin`.

3. Create a simple script `vi hello.m`.

```matlab
function hello
	printf("Hello world!\n");
endfunction
```

4. Make the script executable.

```bash
chmod +x hello.m
```

5. Run the script.

```bash
octave-cli ./hello.m
```

# References

1. https://www.xmodulo.com/how-to-run-octave-script-from-command-line.html