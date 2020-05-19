---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "19 May 2020"
output: html_document
---

## Version
This is the release of the package ``greybox``, v0.6.0

# Test environments
* local ubuntu 19.10, R 3.6.1
* ubuntu 14.04.5 (on travis-ci), R 4.0.0
* win-builder (devel and release)
* rhub

## R CMD check results
R CMD check results
0 errors | 0 warnings | 0 note

## Travis-ci check results
>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

The package is available on CRAN, so it's not clear, what the problem is.

## win-builder
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

## R-hub
**Rhub, Windows Server 2008 R2 SP1, R-devel, 32/64 bit gives an error:**
>* checking package dependencies ... ERROR
>Packages suggested but not available: 'smooth', 'doMC'
>
>Package which this enhances but not available for checking: 'vars'

However, the packages are available on CRAN, so it's not clear, what the problem is.

**Rhub, Ubuntu Linux 16.04 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran:**
>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

Similar to the previous comments. Not clear, why.

**Debian Linux, R-devel, GCC ASAN/UBSAN:**
Finds an error in a function, not related to the greybox package. Also gives a lot of warnings about the methods and functins, which are not used by the package (e.g. "Warning: S3 methods ‘ggplot2::autoplot.zoo’, ‘ggplot2::fortify.zoo’, ‘ggplot2::scale_type.yearmon’, ‘ggplot2::scale_type.yearqtr’ were declared in NAMESPACE but not found").


## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
