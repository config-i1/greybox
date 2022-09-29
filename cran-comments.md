---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "29 September 2022"
output: html_document
---

## Version
This is the release of the package `greybox`, v1.0.6


## Test environments
* local Ubuntu 22.04.1, R 4.2.1
* github actions
* win-builder (devel and release)
* rhub with rhub::check_for_cran() command


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

- Windows latest release with R 4.2.1
- MacOS latest macOS Big Sur 10.16 with R 4.2.1
- Ubuntu 20.04.5 with R 4.2.1


## win-builder
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.


## R-hub
**Windows Server 2022, R-devel, 64 bit:**
> Package suggested but not available: 'doMC'

This is expected, because doMC is not available for Windows.

> Package which this enhances but not available for checking: 'vars'

Not clear, why. `vars` package is on CRAN.

**Rhub, Ubuntu Linux 20.04.1 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran:**
>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

Not clear, why. `vars` package is on CRAN.


## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
