---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "25 October 2018"
output: html_document
---
## Version
This is the release of the package ``greybox``, v0.3.2

## Test environments
* local ubuntu 17.10, R 3.4.4
* ubuntu 14.04.5 (on travis-ci), R 3.5.0
* win-builder (devel and release)

## R CMD check results
R CMD check results
0 errors | 0 warnings | 0 note

## Downstream dependencies
I have also run R CMD check on reverse dependencies of greybox (which only includes smooth package for now). Everything is fine.
Checks on Linux are totally fine (no NOTEs, WARNINGs or ERRORs at all), but Windows gives a NOTE about doMC, which is not available for Windows (for obvious reasons), but is included in suggested packages of greybox.
