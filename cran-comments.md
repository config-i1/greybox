---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "29 March 2020"
output: html_document
---
## Version
This is the release of the package ``greybox``, v0.5.9

# Test environments
* local ubuntu 19.10, R 3.6.1
* ubuntu 14.04.5 (on travis-ci), R 3.6.1
* win-builder (devel and release)

## R CMD check results
R CMD check results
0 errors | 0 warnings | 0 note

## Other checks
Rhub on Windows Server 2008 R2 SP1, R-devel, 32/64 bit; Ubuntu Linux 16.04 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran give the error:
Package which this enhances but not available for checking: 'vars'

However, the package is available on CRAN and there are no notes on my local installations of R or on Travis.


Checks for Windows give (as always) a NOTE about doMC, which is not available for Windows (for obvious reasons). The package is included in suggested of greybox.

## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
