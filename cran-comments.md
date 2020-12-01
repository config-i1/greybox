---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "01 December 2020"
output: html_document
---

## Version
This is the release of the package ``greybox``, v0.6.4

## Test environments
* local ubuntu 19.10, R 4.0.3
* ubuntu 16.04.6 (on travis-ci), R 4.0.0
* win-builder (devel and release)
* rhub

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
>Packages required but not available:
>  'Rcpp'
>Packages suggested but not available:
>  'smooth', 'doMC', 'doParallel', 'foreach', 'testthat', 'rmarkdown',
>  'knitr'
>Package which this enhances but not available for checking: 'vars'
>Packages suggested but not available: 'doMC'

However, the former packages are available on CRAN, so it's not clear, what the problem is. What's wrong with this platform?
`doMC` is just not available for Windows.

**Rhub, Ubuntu Linux 16.04 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran:**
>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

Similar to the previous comments. Not clear, why. Both are on CRAN.

## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
