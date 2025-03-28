% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/alaplace.R
\name{dalaplace}
\alias{dalaplace}
\alias{ALaplace}
\alias{palaplace}
\alias{qalaplace}
\alias{ralaplace}
\title{Asymmetric Laplace Distribution}
\usage{
dalaplace(q, mu = 0, scale = 1, alpha = 0.5, log = FALSE)

palaplace(q, mu = 0, scale = 1, alpha = 0.5)

qalaplace(p, mu = 0, scale = 1, alpha = 0.5)

ralaplace(n = 1, mu = 0, scale = 1, alpha = 0.5)
}
\arguments{
\item{q}{vector of quantiles.}

\item{mu}{vector of location parameters (means).}

\item{scale}{vector of scale parameters.}

\item{alpha}{value of asymmetry parameter. Varies from 0 to 1.}

\item{log}{if \code{TRUE}, then probabilities are returned in
logarithms.}

\item{p}{vector of probabilities.}

\item{n}{number of observations. Should be a single number.}
}
\value{
Depending on the function, various things are returned
(usually either vector or scalar):
\itemize{
\item \code{dalaplace} returns the density function value for the
provided parameters.
\item \code{palaplace} returns the value of the cumulative function
for the provided parameters.
\item \code{qalaplace} returns quantiles of the distribution. Depending
on what was provided in \code{p}, \code{mu} and \code{scale}, this
can be either a vector or a matrix, or an array.
\item \code{ralaplace} returns a vector of random variables
generated from the Laplace distribution. Depending on what was
provided in \code{mu} and \code{scale}, this can be either a vector
or a matrix or an array.
}
}
\description{
Density, cumulative distribution, quantile functions and random number
generation for the Asymmetric Laplace distribution with the
location parameter mu, scale and the asymmetry parameter alpha.
}
\details{
When mu=0 and scale=1, the Laplace distribution becomes standardized.
The distribution has the following density function:

f(x) = alpha (1-alpha) / scale exp(-(x-mu)/scale (alpha - I(x<=mu))),

where I(.) is the indicator function (equal to 1 if the condition is
satisfied and zero otherwise).

When alpha=0.5, then the distribution becomes Symmetric Laplace, where
scale = 1/2 MAE.

This distribution function aligns with the quantile estimates of
parameters (Geraci & Bottai, 2007).

Finally, both \code{palaplace} and \code{qalaplace} are returned for
the lower tail of the distribution.
}
\examples{
x <- dalaplace(c(-100:100)/10, 0, 1, 0.2)
plot(x, type="l")

x <- palaplace(c(-100:100)/10, 0, 1, 0.2)
plot(x, type="l")

qalaplace(c(0.025,0.975), 0, c(1,2), c(0.2,0.3))

x <- ralaplace(1000, 0, 1, 0.2)
hist(x)

}
\references{
\itemize{
\item Geraci Marco, Bottai Matteo (2007). Quantile regression for
longitudinal data using the asymmetric Laplace distribution.
Biostatistics (2007), 8, 1, pp. 140-154
\doi{10.1093/biostatistics/kxj039}
\item Yu, K., & Zhang, J. (2005). A three-parameter asymmetric
laplace distribution and its extension. Communications in Statistics
- Theory and Methods, 34, 1867-1879.
\doi{10.1080/03610920500199018}
}
}
\seealso{
\code{\link[greybox]{Distributions}}
}
\author{
Ivan Svetunkov, \email{ivan@svetunkov.com}
}
\keyword{distribution}
