---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "20 April 2019"
output: html_document
---
## Version
This is the release of the package ``greybox``, v0.5.0

## Test environments
* local ubuntu 18.10, R 3.5.1
* ubuntu 14.04.5 (on travis-ci), R 3.5.3
* win-builder (devel and release)

## R CMD check results
R CMD check results
0 errors | 0 warnings | 0 note

Travis gives a note:
Package which this enhances but not available for checking: ‘vars’

However, the package is available on CRAN and there are no notes on my local installations of R.

## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
Checks for Windows give (as always) a NOTE about doMC, which is not available for Windows (for obvious reasons). The package is included in suggested of greybox.
