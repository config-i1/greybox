---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "12 March 2021"
output: html_document
---

## Version
This is the release of the package ``greybox``, v0.6.8

## Test environments
* local ubuntu 20.04, R 4.0.4
* ubuntu 16.04.6 (on travis-ci), R 4.0.2
* win-builder (devel and release)
* rhub - see the comments below.

## R CMD check results
R CMD check results
0 errors | 0 warnings | 0 note

## win-builder
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.


## R-hub
**Rhub, Windows Server 2008 R2 SP1, R-devel, 32/64 bit gives an error:**
>* checking package dependencies ... ERROR
>Package suggested but not available: 'doMC'
>
>Package which this enhances but not available for checking: 'vars'

'doMC' is not available for Windows. Why 'vars' is not available, is not clear.

**Rhub, Ubuntu Linux 16.04 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran:**
>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

Similar to the previous comments. Not clear, why. Both are on CRAN.

**Debian Linux, R-devel, GCC ASAN/UBSAN**
Fails, complaining that forecast package is not available. Not sure why.

## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
