---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "01 December 2021"
output: html_document
---

## Version
This is the release of the package ``greybox``, v1.0.2

## Test environments
* local ubuntu 20.04.3, R 4.1.2
* github actions
* win-builder (devel and release)
* rhub with rhub::check_for_cran() command

## R CMD check results
> checking installed package size ... NOTE
>    installed size is  5.1Mb
>    sub-directories of 1Mb or more:
>      R      1.1Mb
>      doc    2.5Mb
>      libs   1.1Mb
>
>R CMD check results
>0 errors | 0 warnings | 1 note

## Github actions
Successful checks for:

- Windows latest release with R 4.1.1
- MacOS latest macOS Catalina 10.15.7 with R 4.1.1
- Ubuntu 20.04.3 with R 4.1.2

## win-builder
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

>Found the following (possibly) invalid URLs:
>  URL: https://doi.org/10.2307/2533213
>    From: man/InformationCriteria.Rd
>    Status: 403
>    Message: Forbidden

Not clear why. The paper is available, the url works.

## R-hub
**Windows Server 2008 R2 SP1, R-devel, 32/64 bit:**
> Package suggested but not available: 'doMC'

This is expected, because doMC is not available for Windows.

> Package which this enhances but not available for checking: 'vars'

Not clear, why. `vars` package is on CRAN.

**Rhub, Ubuntu Linux 20.04.1 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran:**
>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

Not clear, why. `vars` package is on CRAN.

### Debian Linux, R-devel, GCC ASAN/UBSAN
> ERROR: dependency ‘httr’ is not available for package ‘texreg’

Not clear why httr is not available. It is on CRAN.

## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
