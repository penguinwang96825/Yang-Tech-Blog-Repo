---
title: Install Docker on Windows without Hyper-V
top: false
cover: false
toc: true
mathjax: true
date: 2022-03-23 12:50:36
img: /images/wallhaven-28wmwg.jpg
coverImg: /images/wallhaven-28wmwg.jpg
summary: Docker is an open-source tool that allows you to run numerous software and operating systems in a containerised environment. The background story is that I wish to run an Android app on Mumu, and this app requires the Hyper-V service to be closed. However, I still need to use Docker to deploy my machine learning project, which requires Hyper-V to be enabled. This puts me in a very difficult position. So in this article, I will try to install and run Docker without using Hyper-V service.
categories: DevOps
tags:
	- Microservices
	- DevOps
---

# Introduction

Docker is an open-source tool that allows you to run numerous software and operating systems in a containerised environment. The background story is that I wish to run an Android app on Mumu, and this app requires the Hyper-V service to be closed. However, I still need to use Docker to deploy my machine learning project, which requires Hyper-V to be enabled. This puts me in a very difficult position. So in this article, I will try to install and run Docker without using Hyper-V service.

# Steps

1. Disable `Virtual Machine Platform`, `Windows Hypervisor Platform` and `Windows Subsystem for Linux` from Windows Features.

<figure>
  <img src="search.png" width=100%>
</figure>

<figure>
  <img src="features.png" width=400>
</figure>

2. Install `Docker ToolBox` executable file. Download the executable binaries from their [Github Page](https://github.com/docker-archive/toolbox/releases).

3. Run the installer via `Docker Toolbox Setup Wizard`. Click on all the **Next** button like installing any other Windows application.

4. Finally, after installing everything you will see an icon on your Desktop, and next run `Docker Quickstart Terminal` Without Hyper-V.

<figure>
  <img src="toolbox.png" width=100%>
</figure>

5. Check the Docker is working or not.

```bash
docker version
```

```
Client:
 Version:           19.03.1
 API version:       1.40
 Go version:        go1.12.7
 Git commit:        74b1e89e8a
 Built:             Wed Jul 31 15:18:18 2019
 OS/Arch:           windows/amd64
 Experimental:      false

Server: Docker Engine - Community
 Engine:
  Version:          19.03.12
  API version:      1.40 (minimum version 1.12)
  Go version:       go1.13.10
  Git commit:       48a66213fe
  Built:            Mon Jun 22 15:49:35 2020
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          v1.2.13
  GitCommit:        7ad184331fa3e55e52b890ea95e65ba581ae3429
 runc:
  Version:          1.0.0-rc10
  GitCommit:        dc9208a3303feef5b3839f4323d9beb36df0a9dd
 docker-init:
  Version:          0.18.0
  GitCommit:        fec3683
```

# References

1. https://www.how2shout.com/how-to/install-docker-without-hyper-v-on-windows-10-7.html
2. https://poweruser.blog/docker-on-windows-10-without-hyper-v-a529897ed1cc
3. https://www.zhihu.com/question/264353707
4. https://mumu.163.com/help/20210511/35041_946700.html?fqbanner
5. https://github.com/docker/for-win/issues/2192