---
title: Build a Dockerfile Include TaLib Wrapper
top: false
cover: false
toc: true
mathjax: true
date: 2022-04-01 02:59:49
img: /images/wallhaven-g7k8x7.jpg
coverImg: /images/wallhaven-g7k8x7.jpg
summary: Install TA-Lib can be pain in the ass on windows 10 or other operating systems, so this article will show you how to build a Dockerfile image including Python TA-Lib wrapper and dependencies. 
categories: DevOps
tags:
	- Docker
	- Python
	- TaLib
---

# Introduction

Install TA-Lib can be pain in the ass on windows 10 or other operating systems, so this article will show you how to build a Dockerfile image including Python TA-Lib wrapper and dependencies. 

# Steps

1. Create a file called `Dockerfile`.

2. Write the following inside `Dockerfile`:

```
FROM continuumio/miniconda3

# Set up the PATH variables
ARG DEBIAN_FRONTEND=noninteractive
ENV TA_LIBRARY_PATH $PREFIX/lib
ENV TA_INCLUDE_PATH $PREFIX/include

# Get the dependencies
RUN apt-get update
RUN apt-get install --assume-yes apt-utils
RUN apt-get -yq install gcc automake autoconf libtool make
RUN apt-get -yq install wget
RUN apt install build-essential -y --no-install-recommends

# Install TA-Lib
RUN /bin/bash -c "wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && pip3 install TA-Lib"
RUN /bin/bash -c "rm -r ta-lib ta-lib-0.4.0-src.tar.gz"
```

3. (optional) If you want to install it inside a conda environment, just add some variables in you PATH environment.

```
FROM continuumio/miniconda3

# Set up the PATH variables
ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_ENV_NAME=talib
ENV TA_LIBRARY_PATH $PREFIX/lib
ENV TA_INCLUDE_PATH $PREFIX/include

# Get the dependencies
RUN apt-get update
RUN apt-get install --assume-yes apt-utils
RUN apt-get -yq install gcc automake autoconf libtool make
RUN apt-get -yq install wget
RUN apt install build-essential -y --no-install-recommends

# Set up conda variables
RUN conda update -n base -c base conda
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/$CONDA_ENV_NAME/bin:$PATH
RUN /bin/bash -c "source activate talib"

# Install TA-Lib
RUN /bin/bash -c "wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && pip3 install TA-Lib"

RUN /bin/bash -c "rm -r ta-lib ta-lib-0.4.0-src.tar.gz"
```

# References

1. https://mrjbq7.github.io/ta-lib/install.html
2. https://gist.github.com/mdalvi/e08115381992e42b43cad861dfe417d2