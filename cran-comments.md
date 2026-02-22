---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "20 February 2026"
output: html_document
---

## Update

Fixed URLs in the README.md


## Version
This is the release of the package `greybox`, v2.0.8.


## Test environments
* local Ubuntu 25.10 R 4.5.1
* github actions
* win-builder (devel and release)
* rhub v2


## R CMD check results
> checking installed package size ... NOTE
>    installed size is  5.1Mb
>    sub-directories of 1Mb or more:
>      R      1.1Mb
>      doc    2.6Mb
>      libs   1.1Mb
>
>R CMD check results
>0 errors | 0 warnings | 1 note


## Github actions
Successful checks for:

- Windows latest release with R 4.5.1
- MacOS latest macOS Monterey 12.6.8 with R 4.5.1
- Ubuntu latest with R 4.5.1


## win-builder
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.


## R-hub
All is fine

## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
