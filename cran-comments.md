---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "05 February 2022"
output: html_document
---

## Version
This is the release of the package `greybox`, v1.0.4

The main reason to submit it is because now it uses `generics` package, importing `forecast` method. This will resolve conflicts with other package.


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
>      doc    2.6Mb
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
**Windows Server 2022, R-devel, 64 bit:**
> Package suggested but not available: 'doMC'

This is expected, because doMC is not available for Windows.

> Package which this enhances but not available for checking: 'vars'

Not clear, why. `vars` package is on CRAN.

>Found the following (possibly) invalid URLs:
>  URL: https://doi.org/10.1029/WR006i002p00505
>    From: man/TPLNormal.Rd
>    Status: 503
>    Message: Service Unavailable

This is wrong. The URL is valid.


**Rhub, Ubuntu Linux 20.04.1 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran:**
>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

Not clear, why. `vars` package is on CRAN.


>Found the following (possibly) invalid URLs:
>  URL: https://doi.org/10.1029/WR006i002p00505
>    From: man/TPLNormal.Rd
>    Status: 503
>    Message: Service Unavailable
>  URL: https://doi.org/10.2307/2533213
>    From: man/InformationCriteria.Rd
>    Status: 403
>    Message: Forbidden

Both URLs are valid, the papers are accessible.


### Debian Linux, R-devel, GCC ASAN/UBSAN
> ERROR: compilation failed for package ‘Rcpp’

Not clear why compilation failed, but this has nothing to do with greybox.


## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
