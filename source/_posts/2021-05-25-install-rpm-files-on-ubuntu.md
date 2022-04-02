---
title: Install RPM Files on Ubuntu
date: 2021-05-25 10:17:00
author: Yang Wang
top: false
cover: false
toc: true
mathjax: true
summary: On Ubuntu, I will show you how to instal RPM packages. The design of Debian-based systems like Ubuntu and RedHat-based systems like CentOS is quite similar. There are, however, a few minor differences. For example, RPM files are used to represent software packages on RedHat-based systems, while DEB files are used on Debian-based systems. An RPM package can be installed in one of two ways, by converting the RPM file to a DEB file or by installing the RPM file directly. Both approaches are easy, but depending on the package being installed, installing RPM packages on a Debian-based system can cause problems.
img: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/25/2021-05-25-install-rpm-files-on-ubuntu/wallhaven-z8p2mg.jpg?raw=true
coverImg: https://github.com/penguinwang96825/penguinwang96825.github.io/blob/master/2021/05/25/2021-05-25-install-rpm-files-on-ubuntu/wallhaven-z8p2mg.jpg?raw=true
categories: Script
tags:
  - Installation
  - RPM
  - Ubuntu
---

# Introduction

On Ubuntu, I'll show you how to instal RPM packages. The design of Debian-based systems like Ubuntu and RedHat-based systems like CentOS is quite similar. There are, however, a few minor differences. For example, RPM files are used to represent software packages on RedHat-based systems, while DEB files are used on Debian-based systems. An RPM package can be installed in one of two ways: by converting the RPM file to a DEB file or by installing the RPM file directly. Both approaches are easy, but depending on the package being installed, installing RPM packages on a Debian-based system can cause problems.

# Procedure

## Step 1: Add the Universe Repository

For the first way, you'll need a package called `Alien`, which converts RPM files to DEB files. You'll need to add a software repository called `Universe` to instal Alien.

```bash
sudo add-apt-repository universe
```

## Step 2: Update apt-get

When the terminal asks for your user account password, type it in. In order for the repository to be used, you must now update `apt-get`.

```bash
sudo apt-get update -y
```

## Step 3: Install Alien Package

Now that the Universe repository has been added, run the following command to install Alien.

```bash
sudo apt-get install alien
```

## Step 4: Convert RPM package to DEB

Once it's up and running, double-check that the software package you downloaded is an RPM file. Go to the folder containing your RPM file. Simply run the command below once you have the RPM file available.

```bash
sudo alien <NameOfPackage>.rpm
```

## Step 5: Install the Converted Package

The file may take a few moments to convert. Once that's done, use `dpkg` to instal the file normally.

```bash
sudo dpkg -i <NameOfPackage>.deb
```

# Example (Connect to University of Sheffield VPN)

I'll show you how to connect an Ubuntu device to a VPN in this article. 

1. Download [Forticlient VPN](https://www.sheffield.ac.uk/polopoly_fs/1.938838!/file/forticlient_vpn_6.4.3.0959_amd64.deb).
2. Convert DEB file to RPM file.
```bash
sudo add-apt-repository universe
sudo apt-get update -y
sudo apt-get install alien
sudo alien forticlient_vpn_6.4.3.0959_x86_64.rpm
sudo dpkg -i forticlient_6.4.3.0959-2_amd64.deb
```
3. Launch Forticlient VPN select Configure VPN.
4. Set VPN Type: SSL VPN.
5. Set Connection Name: UoS-SSL-VPN
6. Set Remote Gateway: remoteaccess.shef.ac.uk
7. Click Save.
8. Select UoS-SSL-VPN as VPN name.
9. Enter your university username - this is the same username you log into MUSE with.
10. Enter your university password - this is the same password you log into MUSE with.

# References

1. https://www.rosehosting.com/blog/how-to-install-rpm-packages-on-ubuntu/
2. https://www.sheffield.ac.uk/it-services/vpn/linux
