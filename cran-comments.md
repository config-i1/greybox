---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "19 May 2021"
output: html_document
---

## Version
This is the release of the package ``greybox``, v0.7.0

## Test environments
* local ubuntu 20.04.2, R 4.0.5
* win-builder (devel and release)
* rhub - see the comments below.

## R CMD check results
R CMD check results
0 errors | 0 warnings | 0 note

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
**Rhub, Windows Server 2008 R2 SP1, R-devel, 32/64 bit gives a PREPERROR:**
> Error: Bioconductor does not yet build and check packages for R version 4.2;
> Execution halted

Something is wrong with Rhub.

**Rhub, Ubuntu Linux 20.04.1 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran:**
>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

Similar to the previous comments. Not clear, why. Both are on CRAN.

**Debian Linux, R-devel, GCC ASAN/UBSAN gives a PREPERROR**
Not clear, what's happening, because the package installs succesfully, tests are done and the end of logs of this reads:
> Finished: SUCCESS

The only error that I can find in the logs is this:
> Error: No such container: greybox_0.7.0.tar.gz-d93e849425864d32b37b2c31ce5bbab7-3

But this looks like an issue with rhub rather than with the package.

## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
