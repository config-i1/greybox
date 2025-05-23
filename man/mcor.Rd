% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mcor.R
\name{mcor}
\alias{mcor}
\title{Multiple correlation}
\usage{
mcor(x, y, use = c("na.or.complete", "complete.obs", "everything",
  "all.obs"))
}
\arguments{
\item{x}{Either data.frame or a matrix}

\item{y}{The numerical variable.}

\item{use}{What observations to use. See \link[stats]{cor} function for details.
The only option that is not available here is \code{"pairwise.complete.obs"}.}
}
\value{
The following list of values is returned:
\itemize{
\item value - The value of the coefficient;
\item statistic - The value of F-statistics associated with the parameter;
\item p.value - The p-value of F-statistics associated with the parameter;
\item df.residual - The number of degrees of freedom for the residuals;
\item df - The number of degrees of freedom for the data.
}
}
\description{
Function calculates multiple correlation between y and x, constructing a linear
regression model
}
\details{
This is based on the linear regression model with the set of variables in x. The
returned value is just a coefficient of multiple correlation from regression,
the F-statistics of the model (thus testing the null hypothesis that all the
parameters are equal to zero), the associated p-value and the degrees of freedom.

See details in the vignette "Marketing analytics with greybox":
\code{vignette("maUsingGreybox","greybox")}
}
\examples{

mcor(mtcars$am, mtcars$mpg)

}
\seealso{
\code{\link[base]{table}, \link[greybox]{tableplot}, \link[greybox]{spread},
\link[greybox]{cramer}, \link[greybox]{association}}
}
\author{
Ivan Svetunkov, \email{ivan@svetunkov.com}
}
\keyword{htest}
