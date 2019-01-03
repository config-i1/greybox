---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "04 January 2019"
output: html_document
---
## Version
This is the release of the package ``greybox``, v0.4.0

## Test environments
* local ubuntu 18.10, R 3.5.2
* ubuntu 14.04.5 (on travis-ci), R 3.5.1
* win-builder (devel and release)

## R CMD check results
R CMD check results
0 errors | 0 warnings | 0 note

## Downstream dependencies
R CMD check on reverse dependencies of greyboxare okay.
Checks for Windows give (as always) a NOTE about doMC, which is not available for Windows (for obvious reasons). The package is included in suggested of greybox.
