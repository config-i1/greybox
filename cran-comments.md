---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "27 June 2021"
output: html_document
---

## Version
This is the release of the package ``greybox``, v1.0.0

## Test environments
* local ubuntu 20.04.2, R 4.1.0
* win-builder (devel and release)
* Github Actions with windows-latest, macOS-latest and ubuntu-20.04
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

Not clear, why. `vars` package is on CRAN.

## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.
