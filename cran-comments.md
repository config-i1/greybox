---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "22 December 2022"
output: html_document
---

## Version
This is the release of the package `greybox`, v1.0.7


## Test environments
* local Ubuntu 22.04.1, R 4.2.2
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

- Windows latest release with R 4.2.2
- MacOS latest macOS Big Sur 10.16 with R 4.2.2
- Ubuntu 20.04.5 with R 4.2.2


## win-builder
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.


## R-hub

**Rhub, Ubuntu Linux 20.04.1 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran:**
>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

Not clear, why. `vars` package is on CRAN.


**Fedora Linux, R-devel, clang, gfortran**
>Error running filter /usr/bin/pandoc-citeproc:
>Filter returned error status 1
>Error: processing vignette 'ro.Rmd' failed with diagnostics:
>pandoc document conversion failed with error 83
>--- failed re-building ‘ro.Rmd’
>
>SUMMARY: processing the following files failed:
>  ‘alm.Rmd’ ‘greybox.Rmd’ ‘maUsingGreybox.Rmd’ ‘ro.Rmd’
>
>Error: Vignette re-building failed.

Looks like an issue with pandoc on the Fedora Linux server. All the vignettes are compiled on other platforms.

>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

Not clear, why. `vars` package is on CRAN.


## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
