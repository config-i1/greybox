% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pcor.R
\name{pcor}
\alias{pcor}
\title{Partial correlations}
\usage{
pcor(x, y = NULL, use = c("na.or.complete", "complete.obs", "everything",
  "all.obs"), method = c("pearson", "spearman", "kendall"))
}
\arguments{
\item{x}{Either data.frame or a matrix with numeric values.}

\item{y}{The numerical variable.}

\item{use}{What observations to use. See \link[stats]{cor} function for details.
The only option that is not available here is \code{"pairwise.complete.obs"}.}

\item{method}{Which method to use for the calculation of the partial correlations.
This can be either Pearson's, Spearman's or Kendall's coefficient. See \link[stats]{cor}
for details.}
}
\value{
The following list of values is returned:
\itemize{
\item{value - Matrix of the coefficients of partial correlations;}
\item{p.value - The p-values for the parameters;}
\item{method - The method used in the calculations.}
}
}
\description{
Function calculates partial correlations between the provided variables
}
\details{
The calculation is done based on multiple linear regressions. The function calculates
them for each pair of variables based on the residuals of linear models of those
variables from the other variables in the dataset.
}
\examples{

pcor(mtcars)

}
\seealso{
\code{\link[greybox]{mcor}, \link[greybox]{cramer}, \link[greybox]{association}}
}
\author{
Ivan Svetunkov, \email{ivan@svetunkov.com}
}
\keyword{htest}
