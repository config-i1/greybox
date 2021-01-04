---
title: "Cran Comments"
author: "Ivan Svetunkov"
date: "04 January 2021"
output: html_document
---

## Version
This is the release of the package ``greybox``, v0.6.5

## Test environments
* local ubuntu 20.04, R 4.0.3
* ubuntu 16.04.6 (on travis-ci), R 4.0.0
* win-builder (devel and release)
* rhub - see the comments below!

## R CMD check results
R CMD check results
0 errors | 0 warnings | 0 note

## win-builder
>* checking package dependencies ... NOTE
>Package suggested but not available for checking: 'doMC'

This is expected, because doMC is not available for Windows.

>** running examples for arch 'i386' ... [45s] NOTE
>Examples with CPU (user + system) or elapsed time > 10s
>     user system elapsed
>alm 23.94   0.08   24.01
>** running examples for arch 'x64' ... [46s] NOTE
>Examples with CPU (user + system) or elapsed time > 10s
>     user system elapsed
>alm 22.71   0.03   22.77

Not sure what has happened, but nothing affecting the speed of the alm() function has been done in 0.6.5. Comparing it via microbenchmark with the version 0.6.4 shows almost no difference.


## R-hub
**Rhub, Windows Server 2008 R2 SP1, R-devel, 32/64 bit gives an error:**
Rhub says that all is ok, but looking through the logs I see:
> Problem closing connection: No space left on device
> Warning in file(name, "wb") :
> cannot open file 'greybox/inst/doc/maUsingGreybox.R': No space left on device
> Error in file(name, "wb") : cannot open the connection
> Execution halted

This implies that the test was not done fully. This is just FYI.
I know that it usually complains about doMC, which is not available for Windows, but the rest is typically fine.

**Rhub, Ubuntu Linux 16.04 LTS, R-release, GCC; Fedora Linux, R-devel, clang, gfortran:**
>* checking package dependencies ... NOTE
>Package which this enhances but not available for checking: ‘vars’

Similar to the previous comments. Not clear, why. Both are on CRAN.

**Debian Linux, R-devel, GCC ASAN/UBSAN**
Fails, complaining that forecast package is not available. Not sure why.

## Downstream dependencies
R CMD check on reverse dependencies of greybox are okay.

## CRAN checks for the greybox version 0.6.4
Looking through CRAN logs, I see that the package fails to install on r-release-windows-ix86+x86_64. This seems to happen because of a weird issue in C++ for arch - x64:
> cc1plus.exe: out of memory allocating 65536 bytes

The only C++ piece of code in the package is a small function that does polynomials multiplication, which hasn't changed for more than a year.
