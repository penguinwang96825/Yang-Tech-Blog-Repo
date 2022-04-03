---
title: Extract RAR File on Mac
top: false
cover: false
toc: true
mathjax: true
date: 2022-04-03 08:54:04
img: /images/wallhaven-rdxypq.jpg
coverImg: /images/wallhaven-rdxypq.jpg
summary: It is super econvenient and easy to use command line tools to do all the stuff you want. Today I'm gonna show you how to extract RAR files on Mac.
categories: Script
tags:
	- Linux
	- CMD
---

# Introduction

It is super econvenient and easy to use command line tools to do all the stuff you want. Today I'm gonna show you how to extract RAR files on Mac.

# Steps

1. Install Homebrew.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install rar package through Homebrew.

```bash
brew install rar
```

3. To use it, just navigate to your file and type in the following.

```bash
unrar x <filename.rar>
```

Or list files via `unrar l <filename.rar>` and extract single file: `unrar e <filename.rar> <subfolder>/<file> <file-location>`.

4. (optional) Another option to extract the files is via 7z library: `brew install p7zip` and then `7z x <filename.rar>`

# References

1. https://superuser.com/questions/52124/how-can-i-extract-rar-files-on-the-mac