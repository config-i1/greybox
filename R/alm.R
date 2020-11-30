#' Augmented Linear Model
#'
#' Function estimates model based on the selected distribution
#'
#' This is a function, similar to \link[stats]{lm}, but using likelihood for the cases
#' of several non-normal distributions. These include:
#' \enumerate{
#' \item \link[stats]{dnorm} - Normal distribution,
#' \item \link[greybox]{dlaplace} - Laplace distribution,
#' \item \link[greybox]{ds} - S-distribution,
#' \item dgnorm - Generalised Normal distribution,
#' \item \link[stats]{dlogis} - Logistic Distribution,
#' \item \link[stats]{dt} - T-distribution,
#' \item \link[greybox]{dalaplace} - Asymmetric Laplace distribution,
#' \item \link[stats]{dlnorm} - Log normal distribution,
#' \item dllaplace - Log Laplace distribution,
#' \item dls - Log S-distribution,
#' \item dlgnorm - Log Generalised Normal distribution,
#' \item \link[greybox]{dfnorm} - Folded normal distribution,
#' \item \link[greybox]{dbcnorm} - Box-Cox normal distribution,
# \item \link[stats]{dchisq} - Chi-Squared Distribution,
#' \item \link[statmod]{dinvgauss} - Inverse Gaussian distribution,
#' \item \link[greybox]{dlogitnorm} - Logit-normal distribution,
#' \item \link[stats]{dbeta} - Beta distribution,
#' \item \link[stats]{dpois} - Poisson Distribution,
#' \item \link[stats]{dnbinom} - Negative Binomial Distribution,
#' \item \link[stats]{plogis} - Cumulative Logistic Distribution,
#' \item \link[stats]{pnorm} - Cumulative Normal distribution.
#' }
#'
#' This function can be considered as an analogue of \link[stats]{glm}, but with the
#' focus on time series. This is why, for example, the function has \code{ar} and
#' \code{i} parameters and produces time series analysis plots with \code{plot(alm(...))}.
#'
#' This function is slower than \code{lm}, because it relies on likelihood estimation
#' of parameters, hessian calculation and matrix multiplication. So think twice when
#' using \code{distribution="dnorm"} here.
#'
#' The estimation is done via the maximisation of likelihood of a selected distribution,
#' so the number of estimated parameters always includes the scale. Thus the number of degrees
#' of freedom of the model in case of \code{alm} will typically be lower than in the case of
#' \code{lm}.
#'
#' See more details and examples in the vignette for "ALM": \code{vignette("alm","greybox")}
#'
#' @template author
#' @template keywords
#'
#' @param formula an object of class "formula" (or one that can be coerced to
#' that class): a symbolic description of the model to be fitted.
#' @param data a data frame or a matrix, containing the variables in the model.
#' @param subset an optional vector specifying a subset of observations to be
#' used in the fitting process.
#' @param na.action	a function which indicates what should happen when the
#' data contain NAs. The default is set by the na.action setting of
#' \link[base]{options}, and is \link[stats]{na.fail} if that is unset. The
#' factory-fresh default is \link[stats]{na.omit}. Another possible value
#' is NULL, no action. Value \link[stats]{na.exclude} can be useful.
#' @param distribution what density function to use in the process. The full
#' name of the distribution should be provided here. Values with "d" in the
#' beginning of the name refer to the density function, while "p" stands for
#' "probability" (cumulative distribution function). The names align with the
#' names of distribution functions in R. For example, see \link[stats]{dnorm}.
#' @param loss The type of Loss Function used in optimization. \code{loss} can
#' be:
#' \itemize{
#' \item \code{likelihood} - the model is estimated via the maximisation of the
#' likelihood of the function specified in \code{distribution};
#' \item \code{MSE} (Mean Squared Error),
#' \item \code{MAE} (Mean Absolute Error),
#' \item \code{HAM} (Half Absolute Moment),
#' \item \code{LASSO} - use LASSO to shrink the parameters of the model;
#' \item \code{RIDGE} - use RIDGE to shrink the parameters of the model;
#' }
#' In case of LASSO / RIDGE, the variables are not normalised prior to the estimation,
#' but the parameters are divided by the standard deviations of explanatory variables
#' inside the optimisation. As the result the parameters of the final model have the
#' same interpretation as in the case of classical linear regression. Note that the
#' user is expected to provide the parameter \code{lambda}.
#'
#' A user can also provide their own function here as well, making sure
#' that it accepts parameters \code{actual}, \code{fitted} and \code{B}. Here is an
#' example:
#'
#' \code{lossFunction <- function(actual, fitted, B, xreg) return(mean(abs(actual-fitted)))}
#' \code{loss=lossFunction}
#'
#' See \code{vignette("alm","greybox")} for some details on losses and distributions.
#' @param occurrence what distribution to use for occurrence variable. Can be
#' \code{"none"}, then nothing happens; \code{"plogis"} - then the logistic
#' regression using \code{alm()} is estimated for the occurrence part;
#' \code{"pnorm"} - then probit is constructed via \code{alm()} for the
#' occurrence part. In both of the latter cases, the formula used is the same
#' as the formula for the sizes. Finally, an "alm" model can be provided and
#' its estimates will be used in the model construction.
#'
#' If this is not \code{"none"}, then the model is estimated
#' in two steps: 1. Occurrence part of the model; 2. Sizes part of the model
#' (excluding zeroes from the data).
#' @param ar the order of AR to include in the model. Only non-seasonal
#' orders are accepted.
# @param i the order of I to include in the model. Only non-seasonal
# orders are accepted.
#' @param parameters vector of parameters of the linear model. When \code{NULL}, it
#' is estimated.
#' @param fast if \code{TRUE}, then the function won't check whether
#' the data has variability and whether the regressors are correlated. Might
#' cause trouble, especially in cases of multicollinearity.
#' @param ... additional parameters to pass to distribution functions. This
#' includes:
#' \itemize{
#' \item \code{alpha} - value for Asymmetric Laplace distribution;
#' \item \code{size} - the size for the Negative Binomial distribution;
#' \item \code{nu} - the number of degrees of freedom for Chi-Squared and Student's t;
#' \item \code{lambda} - the meta parameter for LASSO / RIDGE. Should be between 0 and 1,
#' regulating the strength of shrinkage, where 0 means don't shrink parameters (use MSE)
#' and 1 means shrink everything (ignore MSE);
#' \item \code{lambdaBC} - lambda for Box-Cox transform parameter in case of Box-Cox
#' Normal Distribution.
#' \item \code{FI=TRUE} will make the function also produce Fisher Information
#' matrix, which then can be used to calculated variances of smoothing parameters
#' and initial states of the model. This is used in the \link[stats]{vcov} method;
#' }
#'
#' You can also pass parameters to the optimiser:
#' \enumerate{
#' \item \code{B} - the vector of starting values of parameters for the optimiser,
#' should correspond to the ordering of the explanatory variables;
#' \item \code{algorithm} - the algorithm to use in optimisation
#' (\code{"NLOPT_LN_SBPLX"} by default).
#' \item \code{maxeval} - maximum number of evaluations to carry out (default is 100);
#' \item \code{maxtime} - stop, when the optimisation time (in seconds) exceeds this;
#' \item \code{xtol_rel} - the precision of the optimiser (the default is 1E-6);
#' \item \code{xtol_abs} - the absolute precision of the optimiser (the default is 1E-8);
#' \item \code{ftol_rel} - the stopping criterion in case of the relative change in the loss
#' function (the default is 1E-4);
#' \item \code{ftol_abs} - the stopping criterion in case of the absolute change in the loss
#' function (the default is 0 - not used);
#' \item \code{print_level} - the level of output for the optimiser (0 by default).
#' If equal to 41, then the detailed results of the optimisation are returned.
#' }
#' You can read more about these parameters by running the function
#' \link[nloptr]{nloptr.print.options}.
#'
#' @return Function returns \code{model} - the final model of the class
#' "alm", which contains:
#' \itemize{
#' \item coefficients - estimated parameters of the model,
#' \item FI - Fisher Information of parameters of the model. Returned only when \code{FI=TRUE},
#' \item fitted - fitted values,
#' \item residuals - residuals of the model,
#' \item mu - the estimated location parameter of the distribution,
#' \item scale - the estimated scale parameter of the distribution,
#' \item distribution - distribution used in the estimation,
#' \item logLik - log-likelihood of the model. Only returned, when \code{loss="likelihood"}
#' and in several other special cases of distribution and loss combinations (e.g. \code{loss="MSE"},
#' distribution="dnorm"),
#' \item loss - the type of the loss function used in the estimation,
#' \item lossFunction - the loss function, if the custom is provided by the user,
#' \item lossValue - the value of the loss function,
#' \item df.residual - number of degrees of freedom of the residuals of the model,
#' \item df - number of degrees of freedom of the model,
#' \item call - how the model was called,
#' \item rank - rank of the model,
#' \item data - data used for the model construction,
#' \item occurrence - the occurrence model used in the estimation,
#' \item B - the value of the optimised parameters. Typically, this is a duplicate of coefficients,
#' \item other - the list of all the other parameters either passed to the
#' function or estimated in the process, but not included in the standard output
#' (e.g. \code{alpha} for Asymmetric Laplace).
#' }
#'
#' @seealso \code{\link[greybox]{stepwise}, \link[greybox]{lmCombine},
#' \link[greybox]{xregTransformer}}
#'
#' @examples
#'
#' ### An example with mtcars data and factors
#' mtcars2 <- within(mtcars, {
#'    vs <- factor(vs, labels = c("V", "S"))
#'    am <- factor(am, labels = c("automatic", "manual"))
#'    cyl  <- factor(cyl)
#'    gear <- factor(gear)
#'    carb <- factor(carb)
#' })
#' # The standard model with Log Normal distribution
#' ourModel <- alm(mpg~., mtcars2[1:30,], distribution="dlnorm")
#' summary(ourModel)
#' \dontrun{plot(ourModel)}
#'
#' # Produce predictions with the one sided interval (upper bound)
#' predict(ourModel, mtcars2[-c(1:30),], interval="p", side="u")
#'
#'
#' ### Artificial data for the other examples
#' xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rlaplace(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' # An example with Laplace distribution
#' ourModel <- alm(y~x1+x2, xreg, subset=c(1:80), distribution="dlaplace")
#' summary(ourModel)
#' plot(predict(ourModel,xreg[-c(1:80),]))
#'
#' # And another one with Asymmetric Laplace distribution (quantile regression)
#' # with optimised alpha
#' ourModel <- alm(y~x1+x2, xreg, subset=c(1:80), distribution="dalaplace")
#' summary(ourModel)
#' plot(predict(ourModel,xreg[-c(1:80),]))
#'
#' # An example with AR(1) order
#' ourModel <- alm(y~x1+x2, xreg, subset=c(1:80), distribution="dnorm", ar=1)
#' summary(ourModel)
#' plot(predict(ourModel,xreg[-c(1:80),]))
#'
#' ### Examples with the count data
#' xreg[,1] <- round(exp(xreg[,1]-70),0)
#'
#' # Negative Binomial distribution
#' ourModel <- alm(y~x1+x2, xreg, subset=c(1:80), distribution="dnbinom")
#' summary(ourModel)
#' predict(ourModel,xreg[-c(1:80),],interval="p",side="u")
#'
#' # Poisson distribution
#' ourModel <- alm(y~x1+x2, xreg, subset=c(1:80), distribution="dpois")
#' summary(ourModel)
#' predict(ourModel,xreg[-c(1:80),],interval="p",side="u")
#'
#'
#' ### Examples with binary response variable
#' xreg[,1] <- round(xreg[,1] / (1 + xreg[,1]),0)
#'
#' # Logistic distribution (logit regression)
#' ourModel <- alm(y~x1+x2, xreg, subset=c(1:80), distribution="plogis")
#' summary(ourModel)
#' plot(predict(ourModel,xreg[-c(1:80),],interval="c"))
#'
#' # Normal distribution (probit regression)
#' ourModel <- alm(y~x1+x2, xreg, subset=c(1:80), distribution="pnorm")
#' summary(ourModel)
#' plot(predict(ourModel,xreg[-c(1:80),],interval="p"))
#'
#' @importFrom pracma hessian
#' @importFrom nloptr nloptr
#' @importFrom stats model.frame sd terms model.matrix
#' @importFrom stats dchisq dlnorm dnorm dlogis dpois dnbinom dt dbeta
#' @importFrom stats plogis
#' @importFrom statmod dinvgauss
#' @importFrom forecast Arima
#' @export alm
alm <- function(formula, data, subset, na.action,
                distribution=c("dnorm","dlaplace","ds","dgnorm","dlogis","dt","dalaplace",
                               "dlnorm","dllaplace","dls","dlgnorm","dbcnorm","dfnorm","dinvgauss",
                               "dpois","dnbinom",
                               "dbeta","dlogitnorm",
                               "plogis","pnorm"),
                loss=c("likelihood","MSE","MAE","HAM","LASSO","RIDGE"),
                occurrence=c("none","plogis","pnorm"),
                # scale=NULL,
                ar=0,# i=0,
                parameters=NULL, fast=FALSE, ...){
# Useful stuff for dnbinom: https://scialert.net/fulltext/?doi=ajms.2010.1.15

    # This is a temporary switch off of I(d)
    i <- 0;
    # Create substitute and remove the original data
    dataSubstitute <- substitute(data);

    cl <- match.call();
    # This is needed in order to have a reasonable formula saved, so that there are no issues with it
    cl$formula <- eval(cl$formula);
    distribution <- match.arg(distribution);
    if(is.function(loss)){
        lossFunction <- loss;
        loss <- "custom";
    }
    else{
        lossFunction <- NULL;
        loss <- match.arg(loss);
    }

    #### Functions used in the estimation ####
    ifelseFast <- function(condition, yes, no){
        if(condition){
            return(yes);
        }
        else{
            return(no);
        }
    }

    meanFast <- function(x){
        return(sum(x) / length(x));
    }

    # Function for the Box-Cox transform
    bcTransform <- function(y, lambdaBC){
        if(lambdaBC==0){
            return(log(y));
        }
        else{
            return((y^lambdaBC-1)/lambdaBC);
        }
    }

    # Function for the inverse Box-Cox transform
    bcTransformInv <- function(y, lambdaBC){
        if(lambdaBC==0){
            return(exp(y));
        }
        else{
            return((y*lambdaBC+1)^{1/lambdaBC});
        }
    }

    meanFast <- function(x){
        return(sum(x) / length(x));
    }

    fitter <- function(B, distribution, y, matrixXreg){
        if(distribution=="dalaplace"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- alpha;
            }
        }
        else if(distribution=="dnbinom"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- size;
            }
        }
        else if(distribution=="dchisq"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- nu;
            }
        }
        else if(distribution=="dfnorm"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- sigma;
            }
        }
        else if(any(distribution==c("dgnorm","dlgnorm"))){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- beta;
            }
        }
        else if(distribution=="dbcnorm"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- lambdaBC;
            }
        }
        else if(distribution=="dt"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- nu;
            }
        }
        else{
            other <- NULL;
        }

        # If there is ARI, then calculate polynomials
        if(all(c(arOrder,iOrder)>0)){
            poly1[-1] <- -B[(nVariablesExo+1):nVariables];
            # This condition is needed for cases of only ARI models
            if(nVariables>arOrder){
                B <- c(B[1:nVariablesExo], -polyprod(poly2,poly1)[-1]);
            }
            else{
                B <- -polyprod(poly2,poly1)[-1];
            }
        }
        else if(iOrder>0){
            B <- c(B, -poly2[-1]);
        }
        else if(arOrder>0){
            poly1[-1] <- -B[(nVariablesExo+1):nVariables];
        }

        # This is a hack. If lambda=1, then we only need the mean of the data
        if(any(loss==c("LASSO","RIDGE")) && lambda==1){
            B[-1] <- 0;
        }

        mu[] <- switch(distribution,
                       "dinvgauss"=,
                       "dpois" =,
                       "dnbinom" = exp(matrixXreg %*% B),
                       "dchisq" = ifelseFast(any(matrixXreg %*% B <0),1E+100,(matrixXreg %*% B)^2),
                       "dbeta" = exp(matrixXreg %*% B[1:(length(B)/2)]),
                       "dlogitnorm"=,
                       "dnorm" =,
                       "dlaplace" =,
                       "ds" =,
                       "dgnorm" =,
                       "dlogis" =,
                       "dt" =,
                       "dalaplace" =,
                       "dlnorm" =,
                       "dllaplace" =,
                       "dls" =,
                       "dlgnorm" =,
                       "dbcnorm"=,
                       "dfnorm" =,
                       "pnorm" =,
                       "plogis" = matrixXreg %*% B
        );

        scale <- switch(distribution,
                        "dbeta" = exp(matrixXreg %*% B[-c(1:(length(B)/2))]),
                        "dnorm" = sqrt(sum((y[otU]-mu[otU])^2)/obsInsample),
                        "dlaplace" = sum(abs(y[otU]-mu[otU]))/obsInsample,
                        "ds" = sum(sqrt(abs(y[otU]-mu[otU]))) / (obsInsample*2),
                        "dgnorm" = (other*sum(abs(y[otU]-mu[otU])^other)/obsInsample)^{1/other},
                        "dlogis" = sqrt(sum((y[otU]-mu[otU])^2)/obsInsample * 3 / pi^2),
                        "dalaplace" = sum((y[otU]-mu[otU]) * (other - (y[otU]<=mu[otU])*1))/obsInsample,
                        "dlnorm" = sqrt(sum((log(y[otU])-mu[otU])^2)/obsInsample),
                        "dllaplace" = sum(abs(log(y[otU])-mu[otU]))/obsInsample,
                        "dls" = sum(sqrt(abs(log(y[otU])-mu[otU]))) / (obsInsample*2),
                        "dlgnorm" = (other*sum(abs(log(y[otU])-mu[otU])^other)/obsInsample)^{1/other},
                        "dbcnorm" = sqrt(sum((bcTransform(y[otU],other)-mu[otU])^2)/obsInsample),
                        "dinvgauss" = sum((y[otU]/mu[otU]-1)^2 / (y[otU]/mu[otU]))/obsInsample,
                        "dlogitnorm" = sqrt(sum((log(y[otU]/(1-y[otU]))-mu[otU])^2)/obsInsample),
                        "dfnorm" = abs(other),
                        "dt" = ,
                        "dchisq" =,
                        "dnbinom" = abs(other),
                        "dpois" = mu[otU],
                        "pnorm" = sqrt(meanFast(qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1)^2)),
                        # Here we use the proxy from Svetunkov & Boylan (2019)
                        "plogis" = sqrt(meanFast(log((1 + y * (1 + exp(mu))) /
                                                         (1 + exp(mu) * (2 - y) - y))^2))
        );

        return(list(mu=mu,scale=scale,other=other,poly1=poly1));
    }

    fitterRecursive <- function(B, distribution, y, matrixXreg){
        fitterReturn <- fitter(B, distribution, y, matrixXreg);
        for(i in 1:ariOrder){
            matrixXreg[ariZeroes[,i],nVariablesExo+i] <- switch(distribution,
                                                                "dnbinom" =,
                                                                "dpois" =,
                                                                "dbeta" = log(fitterReturn$mu),
                                                                fitterReturn$mu)[!otU][1:ariZeroesLengths[i]];
            if(distribution=="dbcnorm"){
                matrixXreg[,nVariablesExo+i][!ariZeroes[,i]] <- bcTransform(ariElementsOriginal[!ariZeroes[,i],i],
                                                                            fitterReturn$other);
            }
        }
        # matrixXreg <- fitterRecursion(matrixXreg, B, y, ariZeroes, nVariablesExo, distribution);
        fitterReturn <- fitter(B, distribution, y, matrixXreg);
        fitterReturn$matrixXreg <- matrixXreg;
        return(fitterReturn);
    }

    CF <- function(B, distribution, loss, y, matrixXreg, recursiveModel, denominator){
        if(recursiveModel){
            fitterReturn <- fitterRecursive(B, distribution, y, matrixXreg);
        }
        else{
            fitterReturn <- fitter(B, distribution, y, matrixXreg);
        }

        if(loss=="likelihood"){
            # The original log-likelilhood
            CFValue <- -sum(switch(distribution,
                                   "dnorm" = dnorm(y[otU], mean=fitterReturn$mu[otU], sd=fitterReturn$scale, log=TRUE),
                                   "dlaplace" = dlaplace(y[otU], mu=fitterReturn$mu[otU], scale=fitterReturn$scale, log=TRUE),
                                   "ds" = ds(y[otU], mu=fitterReturn$mu[otU], scale=fitterReturn$scale, log=TRUE),
                                   "dgnorm" = dgnorm(y[otU], mu=fitterReturn$mu[otU], alpha=fitterReturn$scale,
                                                     beta=fitterReturn$other, log=TRUE),
                                   "dlogis" = dlogis(y[otU], location=fitterReturn$mu[otU], scale=fitterReturn$scale, log=TRUE),
                                   "dt" = dt(y[otU]-fitterReturn$mu[otU], df=fitterReturn$scale, log=TRUE),
                                   "dalaplace" = dalaplace(y[otU], mu=fitterReturn$mu[otU], scale=fitterReturn$scale,
                                                           alpha=fitterReturn$other, log=TRUE),
                                   "dlnorm" = dlnorm(y[otU], meanlog=fitterReturn$mu[otU], sdlog=fitterReturn$scale, log=TRUE),
                                   "dllaplace" = dlaplace(log(y[otU]), mu=fitterReturn$mu[otU],
                                                          scale=fitterReturn$scale, log=TRUE)-log(y[otU]),
                                   "dls" = ds(log(y[otU]), mu=fitterReturn$mu[otU], scale=fitterReturn$scale, log=TRUE)-log(y[otU]),
                                   "dlgnorm" = dgnorm(log(y[otU]), mu=fitterReturn$mu[otU], alpha=fitterReturn$scale,
                                                      beta=fitterReturn$other, log=TRUE)-log(y[otU]),
                                   "dbcnorm" = dbcnorm(y[otU], mu=fitterReturn$mu[otU], sigma=fitterReturn$scale,
                                                       lambda=fitterReturn$other, log=TRUE),
                                   "dfnorm" = dfnorm(y[otU], mu=fitterReturn$mu[otU], sigma=fitterReturn$scale, log=TRUE),
                                   "dinvgauss" = dinvgauss(y[otU], mean=fitterReturn$mu[otU],
                                                           dispersion=fitterReturn$scale/fitterReturn$mu[otU], log=TRUE),
                                   "dchisq" = dchisq(y[otU], df=fitterReturn$scale, ncp=fitterReturn$mu[otU], log=TRUE),
                                   "dpois" = dpois(y[otU], lambda=fitterReturn$mu[otU], log=TRUE),
                                   "dnbinom" = dnbinom(y[otU], mu=fitterReturn$mu[otU], size=fitterReturn$scale, log=TRUE),
                                   "dlogitnorm" = dlogitnorm(y[otU], mu=fitterReturn$mu[otU], sigma=fitterReturn$scale, log=TRUE),
                                   "dbeta" = dbeta(y[otU], shape1=fitterReturn$mu[otU], shape2=fitterReturn$scale[otU], log=TRUE),
                                   "pnorm" = c(pnorm(fitterReturn$mu[ot], mean=0, sd=1, log.p=TRUE),
                                               pnorm(fitterReturn$mu[!ot], mean=0, sd=1, lower.tail=FALSE, log.p=TRUE)),
                                   "plogis" = c(plogis(fitterReturn$mu[ot], location=0, scale=1, log.p=TRUE),
                                                plogis(fitterReturn$mu[!ot], location=0, scale=1, lower.tail=FALSE, log.p=TRUE))
            ));

            # The differential entropy for the models with the missing data
            if(occurrenceModel){
                CFValue[] <- CFValue + switch(distribution,
                                              "dnorm" =,
                                              "dfnorm" =,
                                              "dbcnorm" =,
                                              "dlogitnorm" =,
                                              "dlnorm" = obsZero*(log(sqrt(2*pi)*fitterReturn$scale)+0.5),
                                              "dgnorm" =,
                                              "dlgnorm" =obsZero*(1/fitterReturn$other-
                                                                      log(fitterReturn$other /
                                                                              (2*fitterReturn$scale*gamma(1/fitterReturn$other)))),
                                              # "dinvgauss" = 0.5*(obsZero*(log(pi/2)+1+suppressWarnings(log(fitterReturn$scale)))-
                                              #                                 sum(log(fitterReturn$mu[!otU]))),
                                              "dinvgauss" = obsZero*(0.5*(log(pi/2)+1+suppressWarnings(log(fitterReturn$scale)))),
                                              "dlaplace" =,
                                              "dllaplace" =,
                                              "ds" =,
                                              "dls" = obsZero*(2 + 2*log(2*fitterReturn$scale)),
                                              "dalaplace" = obsZero*(1 + log(2*fitterReturn$scale)),
                                              "dlogis" = obsZero*2,
                                              "dt" = obsZero*((fitterReturn$scale+1)/2 *
                                                                  (digamma((fitterReturn$scale+1)/2)-digamma(fitterReturn$scale/2)) +
                                                                  log(sqrt(fitterReturn$scale) * beta(fitterReturn$scale/2,0.5))),
                                              "dchisq" = obsZero*(log(2)*gamma(fitterReturn$scale/2)-
                                                                      (1-fitterReturn$scale/2)*digamma(fitterReturn$scale/2)+
                                                                      fitterReturn$scale/2),
                                              "dbeta" = sum(log(beta(fitterReturn$mu[otU],fitterReturn$scale[otU]))-
                                                                (fitterReturn$mu[otU]-1)*
                                                                (digamma(fitterReturn$mu[otU])-
                                                                     digamma(fitterReturn$mu[otU]+fitterReturn$scale[otU]))-
                                                                (fitterReturn$scale[otU]-1)*
                                                                (digamma(fitterReturn$scale[otU])-
                                                                     digamma(fitterReturn$mu[otU]+fitterReturn$scale[otU]))),
                                              # This is a normal approximation of the real entropy
                                              # "dpois" = sum(0.5*log(2*pi*fitterReturn$scale)+0.5),
                                              # "dnbinom" = obsZero*(log(sqrt(2*pi)*fitterReturn$scale)+0.5),
                                              0
                );
            }
        }
        else{
            ### Fitted values in the scale of the original variable
            yFitted[] <- switch(distribution,
                                "dfnorm" = sqrt(2/pi)*fitterReturn$scale*exp(-fitterReturn$mu^2/(2*fitterReturn$scale^2))+
                                    fitterReturn$mu*(1-2*pnorm(-fitterReturn$mu/fitterReturn$scale)),
                                "dnorm" =,
                                "dgnorm" =,
                                "dinvgauss" =,
                                "dlaplace" =,
                                "dalaplace" =,
                                "dlogis" =,
                                "dt" =,
                                "ds" =,
                                "dpois" =,
                                "dnbinom" = fitterReturn$mu,
                                "dchisq" = fitterReturn$mu + nu,
                                "dlnorm" =,
                                "dllaplace" =,
                                "dls" =,
                                "dlgnorm" = exp(fitterReturn$mu),
                                "dlogitnorm" = exp(fitterReturn$mu)/(1+exp(fitterReturn$mu)),
                                "dbcnorm" = bcTransformInv(fitterReturn$mu,fitterReturn$other),
                                "dbeta" = fitterReturn$mu / (fitterReturn$mu + fitterReturn$scale),
                                "pnorm" = pnorm(fitterReturn$mu, mean=0, sd=1),
                                "plogis" = plogis(fitterReturn$mu, location=0, scale=1)
            );

            if(loss=="MSE"){
                CFValue <- meanFast((y-yFitted)^2);
            }
            else if(loss=="MAE"){
                CFValue <- meanFast(abs(y-yFitted));
            }
            else if(loss=="HAM"){
                CFValue <- meanFast(sqrt(abs(y-yFitted)));
            }
            else if(loss=="LASSO"){
                B[] <- B / denominator;

                if(interceptIsNeeded){
                    CFValue <- (1-lambda) * sqrt(meanFast((y-yFitted)^2)) + lambda * sum(abs(B[-1]))
                }
                else{
                    CFValue <- (1-lambda) * sqrt(meanFast((y-yFitted)^2)) + lambda * sum(abs(B))
                }
                # This is a hack. If lambda=1, then we only need the mean of the data
                if(lambda==1){
                    CFValue <- sqrt(meanFast((y-yFitted)^2));
                }
            }
            else if(loss=="RIDGE"){
                B[] <- B / denominator;

                if(interceptIsNeeded){
                    CFValue <- (1-lambda) * sqrt(meanFast((y-yFitted)^2)) + lambda * sqrt(sum(B[-1]^2))
                }
                else{
                    CFValue <- (1-lambda) * sqrt(meanFast((y-yFitted)^2)) + lambda * sqrt(sum(B^2))
                }
                # This is a hack. If lambda=1, then we only need the mean of the data
                if(lambda==1){
                    CFValue <- sqrt(meanFast((y-yFitted)^2));
                }
            }
            else if(loss=="custom"){
                CFValue <- lossFunction(actual=y,fitted=yFitted,B=B,xreg=matrixXreg);
            }
        }

        if(is.nan(CFValue) || is.na(CFValue) || is.infinite(CFValue)){
            CFValue[] <- 1E+300;
        }

        # Check the roots of polynomials
        if(arOrder>0 && any(abs(polyroot(fitterReturn$poly1))<1)){
            CFValue[] <- CFValue / min(abs(polyroot(fitterReturn$poly1)));
        }

        return(CFValue);
    }

    #### Define the rest of parameters ####
    ellipsis <- list(...);

    # Parameters for distributions
    if(distribution=="dalaplace"){
        if(is.null(ellipsis$alpha)){
            ellipsis$alpha <- 0.5
            aParameterProvided <- FALSE;
        }
        else{
            alpha <- ellipsis$alpha;
            aParameterProvided <- TRUE;
        }
    }
    else if(distribution=="dchisq"){
        if(is.null(ellipsis$nu)){
            aParameterProvided <- FALSE;
        }
        else{
            nu <- ellipsis$nu;
            aParameterProvided <- TRUE;
        }
    }
    else if(distribution=="dnbinom"){
        if(is.null(ellipsis$size)){
            aParameterProvided <- FALSE;
        }
        else{
            size <- ellipsis$size;
            aParameterProvided <- TRUE;
        }
    }
    else if(distribution=="dfnorm"){
        if(is.null(ellipsis$sigma)){
            aParameterProvided <- FALSE;
        }
        else{
            sigma <- ellipsis$sigma;
            aParameterProvided <- TRUE;
        }
    }
    else if(any(distribution==c("dgnorm","dlgnorm"))){
        if(is.null(ellipsis$beta)){
            aParameterProvided <- FALSE;
        }
        else{
            beta <- ellipsis$beta;
            aParameterProvided <- TRUE;
        }
    }
    else if(distribution=="dbcnorm"){
        if(is.null(ellipsis$lambdaBC)){
            aParameterProvided <- FALSE;
        }
        else{
            lambdaBC <- ellipsis$lambdaBC;
            aParameterProvided <- TRUE;
        }
    }
    else if(distribution=="dt"){
        if(is.null(ellipsis$nu)){
            aParameterProvided <- FALSE;
        }
        else{
            nu <- ellipsis$nu;
            aParameterProvided <- TRUE;
        }
    }
    else{
        aParameterProvided <- TRUE;
    }
    if(!aParameterProvided && loss!="likelihood"){
        warning("The chosen loss function does not allow optimisation of additional parameters ",
                "for the distribution=\"",distribution,"\". Use likelihood instead. We will use 0.5.",
                call.=FALSE);
            alpha <- nu <- size <- sigma <- beta <- lambdaBC <- nu <- 0.5;
            aParameterProvided <- TRUE;
    }
    # LASSO / RIDGE loss
    if(any(loss==c("LASSO","RIDGE"))){
        warning(paste0("Please, keep in mind that loss='",loss,
                       "' is an experimental option. It might not work correctly."), call.=FALSE);
        if(is.null(ellipsis$lambda)){
            warning("You have not provided lambda parameter. We will set it to zero.", call.=FALSE);
            lambda <- 0;
        }
        else{
            lambda <- ellipsis$lambda;
        }
    }

    # Fisher Information
    if(is.null(ellipsis$FI)){
        FI <- FALSE;
    }
    else{
        FI <- ellipsis$FI;
    }

    # Starting values for the optimiser
    if(is.null(ellipsis$B)){
        B <- NULL;
    }
    else{
        B <- ellipsis$B;
    }
    # Parameters for the nloptr from the ellipsis
    if(is.null(ellipsis$xtol_rel)){
        xtol_rel <- 1E-6;
    }
    else{
        xtol_rel <- ellipsis$xtol_rel;
    }
    if(is.null(ellipsis$algorithm)){
        # if(recursiveModel){
        # algorithm <- "NLOPT_LN_BOBYQA";
        # }
        # else{
        algorithm <- "NLOPT_LN_SBPLX";
        # }
    }
    else{
        algorithm <- ellipsis$algorithm;
    }
    if(is.null(ellipsis$maxtime)){
        maxtime <- -1;
    }
    else{
        maxtime <- ellipsis$maxtime;
    }
    if(is.null(ellipsis$xtol_abs)){
        xtol_abs <- 1E-8;
    }
    else{
        xtol_abs <- ellipsis$xtol_abs;
    }
    if(is.null(ellipsis$ftol_rel)){
        ftol_rel <- 1E-4;
        # LASSO / RIDGE need more accurate estimation
        if(any(loss==c("LASSO","RIDGE"))){
            ftol_rel <- 1e-8;
        }
    }
    else{
        ftol_rel <- ellipsis$ftol_rel;
    }
    if(is.null(ellipsis$ftol_abs)){
        ftol_abs <- 0;
    }
    else{
        ftol_abs <- ellipsis$ftol_abs;
    }
    if(is.null(ellipsis$print_level)){
        print_level <- 0;
    }
    else{
        print_level <- ellipsis$print_level;
    }


    # If occurrence is not provideded, then set it to "none"
    if(is.null(occurrence)){
        occurrence <- "none";
    }
    # See if the occurrence model is provided, and whether we need to treat the data as intermittent
    if(is.occurrence(occurrence)){
        occurrenceModel <- TRUE;
        occurrenceProvided <- TRUE;
    }
    else{
        occurrence <- occurrence[1];
        occurrenceProvided <- FALSE;
        if(all(occurrence!=c("none","plogis","pnorm"))){
            warning(paste0("Sorry, but we don't know what to do with the occurrence '",occurrence,
                        "'. Switching to 'none'."), call.=FALSE);
            occurrence <- "none";
        }

        if(any(occurrence==c("plogis","pnorm"))){
            occurrenceModel <- TRUE;
        }
        else{
            occurrenceModel <- FALSE;
            occurrence <- NULL;
        }
    }

    arOrder <- ar;
    iOrder <- i;
    # Check AR, I and form ARI order
    if(length(arOrder)>1){
        warning("ar must be a scalar, not a vector. Using the first value.", call.=FALSE);
        arOrder <- arOrder[1];
    }
    if(length(iOrder)>1){
        warning("i must be a scalar, not a vector. Using the first value.", call.=FALSE);
        iOrder <- iOrder[1];
    }
    ariOrder <- arOrder + iOrder;
    ariModel <- ifelseFast(ariOrder>0, TRUE, FALSE);
    # Create polynomials for the i and ar orders
    if(iOrder>0){
        poly2 <- c(1,-1);
        if(iOrder>1){
            for(j in 1:(iOrder-1)){
                poly2 <- polyprod(poly2,c(1,-1));
            }
        }
    }
    if(arOrder>0){
        poly1 <- rep(1,arOrder+1);
    }
    else{
        poly1 <- c(1,1);
    }

    #### Form the necessary matrices ####
    # Call similar to lm in order to form appropriate data.frame
    mf <- match.call(expand.dots = FALSE);
    m <- match(c("formula", "data", "subset", "na.action"), names(mf), 0L);
    mf <- mf[c(1L, m)];
    mf$drop.unused.levels <- TRUE;
    mf[[1L]] <- quote(stats::model.frame);

    # If data is provided explicitly, check it
    if(exists("data",inherits=FALSE,mode="numeric") || exists("data",inherits=FALSE,mode="list")){
        if(!is.data.frame(data)){
            data <- as.data.frame(data);
        }
        else{
            dataOrders <- unlist(lapply(data,is.ordered));
            # If there is an ordered factor, remove the bloody ordering!
            if(any(dataOrders)){
                data[dataOrders] <- lapply(data[dataOrders],function(x) factor(x, levels=levels(x), ordered=FALSE));
            }
        }
        mf$data <- data;
        rm(data);

        # If there are NaN values, remove the respective observations
        if(any(sapply(mf$data,is.nan))){
            warning("There are NaN values in the data. This might cause problems. Removing these observations.", call.=FALSE);
            NonNaNValues <- !apply(sapply(mf$data,is.nan),1,any);
            # If subset was not provided, change it
            if(is.null(mf$subset)){
                mf$subset <- NonNaNValues
            }
            else{
                mf$subset <- NonNaNValues & mf$subset;
            }
            dataContainsNaNs <- TRUE;
        }
        else{
            dataContainsNaNs <- FALSE;
        }

        # If there are spaces in names, give a warning
        if(any(grepl("[^A-Za-z0-9,;._-]", all.vars(formula)))){
            warning("The names of your variables contain special characters ",
                    "(such as spaces, comas, brackets etc). alm() might not work properly. ",
                    "It is recommended to use `make.names()` function to fix the names of variables.",
                    call.=FALSE);
            formula <- as.formula(paste0(gsub(paste0("`",all.vars(formula)[1],"`"),
                                              make.names(all.vars(formula)[1]),
                                              all.vars(formula)[1]),"~",
                                         paste0(mapply(gsub, paste0("`",all.vars(formula)[-1],"`"),
                                                       make.names(all.vars(formula)[-1]),
                                                       labels(terms(formula))),
                                                collapse="+")));
            mf$formula <- formula;
        }
        # Fix names of variables
        colnames(mf$data) <- make.names(colnames(mf$data), unique=TRUE);
    }
    else{
        dataContainsNaNs <- FALSE;
    }

    responseName <- all.vars(formula)[1];

    # Make recursive fitted for missing values in case of occurrence model.
    recursiveModel <- occurrenceModel && ariModel;
    # In case of plogis and pnorm, all the ARI values need to be refitted
    if(any(distribution==c("plogis","pnorm")) && ariModel){
        recursiveModel <- TRUE;
    }

    #### Define what to do with the maxeval ####
    if(is.null(ellipsis$maxeval)){
        if(any(distribution==c("dchisq","dpois","dnbinom","dbcnorm","plogis","pnorm")) || recursiveModel){
            maxeval <- 500;
        }
        # The following ones don't really need the estimation. This is for consistency only
        else if(any(distribution==c("dnorm","dlnorm","dlogitnorm")) & !recursiveModel && any(loss==c("likelihood","MSE"))){
            maxeval <- 1;
        }
        else{
            maxeval <- 200;
        }
        # LASSO / RIDGE need more iterations to converge
        if(any(loss==c("LASSO","RIDGE"))){
            maxeval <- 1000;
        }
    }
    else{
        maxeval <- ellipsis$maxeval;
    }

    dataWork <- eval(mf, parent.frame());
    y <- dataWork[,1];

    interceptIsNeeded <- attr(terms(dataWork),"intercept")!=0;
    # Create a model from the provided stuff. This way we can work with factors
    dataWork <- model.matrix(dataWork,data=dataWork);
    obsInsample <- nrow(dataWork);

    if(interceptIsNeeded){
        variablesNames <- colnames(dataWork)[-1];
        matrixXreg <- as.matrix(dataWork[,-1,drop=FALSE]);
        # Include response to the data
        # dataWork <- cbind(y,dataWork[,-1,drop=FALSE]);
    }
    else{
        variablesNames <- colnames(dataWork);
        matrixXreg <- dataWork;
        # Include response to the data
        # dataWork <- cbind(y,dataWork);
        warning("You have asked not to include intercept in the model. We will try to fit the model, ",
                "but this is a very naughty thing to do, and we cannot guarantee that it will work...", call.=FALSE);
    }
    # colnames(dataWork) <- c(responseName, variablesNames);
    rm(dataWork);

    nVariables <- length(variablesNames);
    colnames(matrixXreg) <- variablesNames;

    # Record the subset used in the model
    if(is.null(mf$subset)){
        subset <- rep(TRUE, obsInsample);
    }
    else{
        if(dataContainsNaNs){
            subset <- mf$subset[NonNaNValues];
        }
        else{
            subset <- mf$subset;
        }
    }

    mu <- vector("numeric", obsInsample);
    yFitted <- vector("numeric", obsInsample);
    errors <- vector("numeric", obsInsample);
    ot <- vector("logical", obsInsample);

    if(any(y<0) & any(distribution==c("dfnorm","dlnorm","dllaplace","dls","dbcnorm","dinvgauss","dchisq","dpois","dnbinom"))){
        stop(paste0("Negative values are not allowed in the response variable for the distribution '",distribution,"'"),
             call.=FALSE);
    }

    if(any(y==0) & any(distribution==c("dinvgauss")) & !occurrenceModel){
        stop(paste0("Zero values are not allowed in the response variable for the distribution '",distribution,"'"),
             call.=FALSE);
    }

    if(any(distribution==c("dpois","dnbinom"))){
        if(any(y!=trunc(y))){
            stop(paste0("Count data is needed for the distribution '",distribution,"', but you have fractional numbers. ",
                        "Maybe you should try some other distribution?"),
                 call.=FALSE);
        }
    }

    if(any(distribution==c("dbeta","dlogitnorm"))){
        if(any((y>1) | (y<0))){
            stop(paste0("The response variable should lie between 0 and 1 in distribution=\"",
                        distribution,"\""), call.=FALSE);
        }
        else if(any(y==c(0,1))){
            warning(paste0("The response variable contains boundary values (either zero or one). ",
                           "distribution=\"",distribution,"\" is not estimable in this case. ",
                           "So we used a minor correction for it, in order to overcome this limitation."), call.=FALSE);
            y <- y*(1-2*1e-10);
        }
    }

    if(any(distribution==c("plogis","pnorm"))){
        CDF <- TRUE;
    }
    else{
        CDF <- FALSE;
    }

    if(CDF && any(y!=0 & y!=1)){
        y[] <- (y!=0)*1;
    }

    # Vector with logical for the non-zero observations
    if(CDF || occurrenceModel){
        ot[] <- y!=0;
    }
    else{
        ot[] <- rep(TRUE,obsInsample);
    }

    # Set the number of zeroes and non-zeroes, depending on the type of the model
    if(occurrenceModel){
        # otU is the vector of {0,1} for the occurrence model, not for the CDF!
        otU <- ot;
        obsNonZero <- sum(ot);
        obsZero <- sum(!ot);
    }
    else{
        otU <- rep(TRUE,obsInsample);
        obsNonZero <- obsInsample;
        obsZero <- 0;
    }

    if(!fast){
        #### Checks of the exogenous variables ####
        # Remove the data for which sd=0
        noVariability <- vector("logical",nVariables);
        noVariability[] <- apply((matrixXreg==matrix(matrixXreg[1,],obsInsample,nVariables,byrow=TRUE))[otU,,drop=FALSE],2,all);
        if(any(noVariability)){
            if(all(noVariability)){
                warning("None of exogenous variables has variability. Fitting the straight line.",
                        call.=FALSE);
                matrixXreg <- matrix(1,obsInsample,1);
                nVariables <- 1;
                variablesNames <- "(Intercept)";
            }
            else{
                if(!occurrenceModel && !CDF){
                    warning("Some exogenous variables did not have any variability. We dropped them out.",
                            call.=FALSE);
                }
                matrixXreg <- matrixXreg[,!noVariability,drop=FALSE];
                nVariables <- ncol(matrixXreg);
                variablesNames <- variablesNames[!noVariability];
            }
        }

        # Check the multicollinearity. Don't do it for LASSO / RIDGE
        if(all(loss!=c("LASSO","RIDGE"))){
            corThreshold <- 0.999;
            if(nVariables>1){
                # Check perfectly correlated cases
                corMatrix <- cor(matrixXreg[otU,,drop=FALSE]);
                corHigh <- upper.tri(corMatrix) & abs(corMatrix)>=corThreshold;
                if(any(corHigh)){
                    removexreg <- unique(which(corHigh,arr.ind=TRUE)[,1]);
                    matrixXreg <- matrixXreg[,-removexreg,drop=FALSE];
                    nVariables <- ncol(matrixXreg);
                    variablesNames <- colnames(matrixXreg);
                    if(!occurrenceModel && !CDF){
                        warning("Some exogenous variables were perfectly correlated. We've dropped them out.",
                                call.=FALSE);
                    }
                }
            }

            # Do these checks only when intercept is needed. Otherwise in case of dummies this might cause chaos
            if(nVariables>1 & interceptIsNeeded){
                # Check dummy variables trap
                detHigh <- suppressWarnings(determination(matrixXreg[otU,,drop=FALSE]))>=corThreshold;
                if(any(detHigh)){
                    while(any(detHigh)){
                        removexreg <- which(detHigh>=corThreshold)[1];
                        matrixXreg <- matrixXreg[,-removexreg,drop=FALSE];
                        nVariables <- ncol(matrixXreg);
                        variablesNames <- colnames(matrixXreg);

                        detHigh <- suppressWarnings(determination(matrixXreg))>=corThreshold;
                    }
                    if(!occurrenceModel){
                        warning("Some combinations of exogenous variables were perfectly correlated. We've dropped them out.",
                                call.=FALSE);
                    }
                }
            }
        }
    }

    #### Finish forming the matrix of exogenous variables ####
    # Remove the redudant dummies, if there are any
    varsToLeave <- apply(matrixXreg[otU,,drop=FALSE],2,var)!=0;
    matrixXreg <- matrixXreg[,varsToLeave,drop=FALSE];
    variablesNames <- variablesNames[varsToLeave];
    nVariables <- length(variablesNames);

    if(interceptIsNeeded){
        matrixXreg <- cbind(1,matrixXreg);
        variablesNames <- c("(Intercept)",variablesNames);
        colnames(matrixXreg) <- variablesNames;

        # Check, if redundant dummies are left. Remove the first if this is the case
        # Don't do the check for LASSO / RIDGE
        if(is.null(parameters) && !fast && all(loss!=c("LASSO","RIDGE"))){
            determValues <- suppressWarnings(determination(matrixXreg[otU, -1, drop=FALSE]));
            determValues[is.nan(determValues)] <- 0;
            if(any(determValues==1)){
                matrixXreg <- matrixXreg[,-(which(determValues==1)[1]+1),drop=FALSE];
                variablesNames <- colnames(matrixXreg);
            }
            nVariables <- length(variablesNames);
        }
    }
    # The number of exogenous variables (no ARI elements)
    nVariablesExo <- nVariables;

    #### The model for the scale ####
    # if(!is.null(scale)){
    # }

    #### Estimate parameters of the model ####
    if(is.null(parameters)){
        #### Add AR and I elements in the regression ####
        # This is only done, if the regression is estimated. In the other cases it will already have the expanded values
        if(ariModel){
            # In case of plogis and pnorm, the AR elements need to be generated from a model, i.e. oes from smooth.
            if(any(distribution==c("plogis","pnorm"))){
                if(!requireNamespace("smooth", quietly = TRUE)){
                    yNew <- abs(fitted(Arima(y, order=c(0,1,1))));
                    yNew[is.na(yNew)] <- min(yNew);
                    yNew[yNew==0] <- 1E-10;
                    yNew[] <- log(yNew / (1-yNew));
                    yNew[is.infinite(yNew) & yNew>0] <- max(yNew[is.finite(yNew)]);
                    yNew[is.infinite(yNew) & yNew<0] <- min(yNew[is.finite(yNew)]);
                }
                else{
                    yNew <- smooth::oes(y, occurrence="i", model="MNN", h=1)$fittedModel
                }
                ariElements <- xregExpander(yNew, lags=-c(1:ariOrder), gaps="auto")[,-1,drop=FALSE];
                ariZeroes <- matrix(TRUE,nrow=obsInsample,ncol=ariOrder);
                for(i in 1:ariOrder){
                    ariZeroes[(1:i),i] <- FALSE;
                }
                ariZeroesLengths <- apply(ariZeroes, 2, sum);
            }
            else{
                ariElements <- xregExpander(y, lags=-c(1:ariOrder), gaps="auto")[,-1,drop=FALSE];
            }

            # Get rid of "ts" class
            class(ariElements) <- "matrix";
            ariNames <- paste0(responseName,"Lag",c(1:ariOrder));
            ariTransformedNames <- ariNames;
            colnames(ariElements) <- ariNames;

            # Non-zero sequencies for the recursion mechanism of ar
            if(occurrenceModel){
                ariZeroes <- ariElements == 0;
                ariZeroesLengths <- apply(ariZeroes, 2, sum);
            }

            if(ar>0){
                arNames <- paste0(responseName,"Lag",c(1:ar));
                variablesNames <- c(variablesNames,arNames);
            }
            else{
                arNames <- vector("character",0);
            }
            nVariables <- nVariables + arOrder;
            # Write down the values for the matrixXreg in the necessary transformations
            if(any(distribution==c("dlnorm","dllaplace","dls","dpois","dnbinom"))){
                if(any(y[otU]==0)){
                    # Use Box-Cox if there are zeroes
                    ariElements[] <- bcTransform(ariElements,0.01);
                    ariTransformedNames <- paste0(ariNames,"Box-Cox");
                    colnames(ariElements) <- ariTransformedNames;
                }
                else{
                    ariElements[ariElements<0] <- 0;
                    ariElements[] <- suppressWarnings(log(ariElements));
                    ariElements[is.infinite(ariElements)] <- 0;
                    ariTransformedNames <- paste0(ariNames,"Log");
                    colnames(ariElements) <- ariTransformedNames;
                }
            }
            else if(distribution=="dbcnorm"){
                ariElementsOriginal <- ariElements;
                ariElements[] <- bcTransform(ariElements,0.1);
                ariTransformedNames <- paste0(ariNames,"Box-Cox");
                colnames(ariElements) <- ariTransformedNames;
            }
            else if(distribution=="dchisq"){
                ariElements[] <- sqrt(ariElements);
                ariTransformedNames <- paste0(ariNames,"Sqrt");
                colnames(ariElements) <- ariTransformedNames;
            }

            # Fill in zeroes with the mean values
            ariElements[ariElements==0] <- mean(ariElements[ariElements[,1]!=0,1]);

            matrixXreg <- cbind(matrixXreg, ariElements);
            # dataWork <- cbind(dataWork, ariElements);
        }

        # Set bounds for B as NULL. Then amend if needed
        BLower <- NULL;
        BUpper <- NULL;
        if(is.null(B)){
            #### I(0) initialisation ####
            if(iOrder==0){
                if(any(distribution==c("dlnorm","dllaplace","dls","dlgnorm","dpois","dnbinom","dinvgauss"))){
                    if(any(y[otU]==0)){
                        # Use Box-Cox if there are zeroes
                        B <- .lm.fit(matrixXreg[otU,,drop=FALSE],bcTransform(y[otU],0.01))$coefficients;
                    }
                    else{
                        B <- .lm.fit(matrixXreg[otU,,drop=FALSE],log(y[otU]))$coefficients;
                    }
                }
                else if(any(distribution==c("plogis","pnorm"))){
                    # Box-Cox transform in order to get meaningful initials
                    B <- .lm.fit(matrixXreg,bcTransform(y[otU],0.01))$coefficients;
                }
                else if(distribution=="dbcnorm"){
                    if(!aParameterProvided){
                        B <- c(0.1,.lm.fit(matrixXreg[otU,,drop=FALSE],bcTransform(y[otU],0.1))$coefficients);
                    }
                    else{
                        B <- c(.lm.fit(matrixXreg[otU,,drop=FALSE],bcTransform(y[otU],lambdaBC))$coefficients);
                    }
                }
                else if(distribution=="dbeta"){
                    # In Beta we set B to be twice longer, using first half of parameters for shape1, and the second for shape2
                    # Transform y, just in case, to make sure that it does not hit boundary values
                    B <- .lm.fit(matrixXreg[otU,,drop=FALSE],log(y[otU]/(1-y[otU])))$coefficients;
                    B <- c(B, -B);
                }
                else if(distribution=="dlogitnorm"){
                    B <- .lm.fit(matrixXreg[otU,,drop=FALSE],log(y[otU]/(1-y[otU])))$coefficients;
                }
                else if(distribution=="dchisq"){
                    B <- .lm.fit(matrixXreg[otU,,drop=FALSE],sqrt(y[otU]))$coefficients;
                    if(aParameterProvided){
                        BLower <- rep(-Inf,length(B));
                        BUpper <- rep(Inf,length(B));
                    }
                    else{
                        B <- c(1, B);
                        BLower <- c(0,rep(-Inf,length(B)-1));
                        BUpper <- rep(Inf,length(B));
                    }
                }
                else if(distribution=="dt"){
                    B <- .lm.fit(matrixXreg[otU,,drop=FALSE],y[otU])$coefficients;
                    if(aParameterProvided){
                        BLower <- rep(-Inf,length(B));
                        BUpper <- rep(Inf,length(B));
                    }
                    else{
                        B <- c(2, B);
                        BLower <- c(0,rep(-Inf,length(B)-1));
                        BUpper <- rep(Inf,length(B));
                    }
                }
                else{
                    B <- .lm.fit(matrixXreg[otU,,drop=FALSE],y[otU])$coefficients;
                    BLower <- -Inf;
                    BUpper <- Inf;
                }
            }
            #### I(d) initialisation ####
            # If this is an I(d) model, do the primary estimation in differences
            else{
                # Matrix without the first D rows and without the last D columns
                # matrixXregForDiffs <- matrixXreg[otU,-(nVariables+1:iOrder),drop=FALSE][-c(1:iOrder),,drop=FALSE];

                # Use only AR elements of the matrix, take differences for the initialisation purposes
                matrixXregForDiffs <- matrixXreg[otU,-(nVariables+1:iOrder),drop=FALSE];
                if(arOrder>0){
                    matrixXregForDiffs[-c(1:iOrder),nVariablesExo+c(1:arOrder)] <- diff(matrixXregForDiffs[,nVariablesExo+c(1:arOrder)],
                                                                                        differences=iOrder);
                    matrixXregForDiffs <- matrixXregForDiffs[-c(1:iOrder),,drop=FALSE];
                    matrixXregForDiffs[c(1:iOrder),nVariablesExo+c(1:arOrder)] <- colMeans(matrixXregForDiffs[,nVariablesExo+c(1:arOrder), drop=FALSE]);
                }
                else{
                    matrixXregForDiffs <- matrixXregForDiffs[-c(1:iOrder),,drop=FALSE];
                }

                if(any(distribution==c("dlnorm","dllaplace","dls","dlgnorm","dpois","dnbinom","dinvgauss"))){
                    B <- .lm.fit(matrixXregForDiffs,diff(log(y[otU]),differences=iOrder))$coefficients;
                }
                else if(any(distribution==c("plogis","pnorm"))){
                    # Box-Cox transform in order to get meaningful initials
                    B <- .lm.fit(matrixXregForDiffs,diff(bcTransform(y[otU],0.01),differences=iOrder))$coefficients;
                }
                else if(distribution=="dbcnorm"){
                    if(!aParameterProvided){
                        B <- c(0.1,.lm.fit(matrixXregForDiffs,diff(bcTransform(y[otU],0.1),differences=iOrder))$coefficients);
                    }
                    else{
                        B <- c(.lm.fit(matrixXregForDiffs,diff(bcTransform(y[otU],0.1),differences=iOrder))$coefficients);
                    }
                }
                else if(distribution=="dbeta"){
                    # In Beta we set B to be twice longer, using first half of parameters for shape1, and the second for shape2
                    # Transform y, just in case, to make sure that it does not hit boundary values
                    B <- .lm.fit(matrixXregForDiffs,
                                 diff(log(y/(1-y)),differences=iOrder)[c(1:nrow(matrixXregForDiffs))])$coefficients;
                    B <- c(B, -B);
                }
                else if(distribution=="dlogitnorm"){
                    B <- .lm.fit(matrixXregForDiffs,
                                 diff(log(y/(1-y)),differences=iOrder)[c(1:nrow(matrixXregForDiffs))])$coefficients;
                }
                else if(distribution=="dchisq"){
                    B <- .lm.fit(matrixXregForDiffs,diff(sqrt(y[otU]),differences=iOrder))$coefficients;
                    if(aParameterProvided){
                        BLower <- rep(-Inf,length(B));
                        BUpper <- rep(Inf,length(B));
                    }
                    else{
                        B <- c(1, B);
                        BLower <- c(0,rep(-Inf,length(B)-1));
                        BUpper <- rep(Inf,length(B));
                    }
                }
                else if(distribution=="dt"){
                    B <- .lm.fit(matrixXregForDiffs,diff(y[otU],differences=iOrder))$coefficients;
                    if(aParameterProvided){
                        BLower <- rep(-Inf,length(B));
                        BUpper <- rep(Inf,length(B));
                    }
                    else{
                        B <- c(2, B);
                        BLower <- c(0,rep(-Inf,length(B)-1));
                        BUpper <- rep(Inf,length(B));
                    }
                }
                else{
                    B <- .lm.fit(matrixXregForDiffs,diff(y[otU],differences=iOrder))$coefficients;
                    BLower <- -Inf;
                    BUpper <- Inf;
                }
            }

            if(distribution=="dnbinom"){
                if(!aParameterProvided){
                    B <- c(var(y[otU]), B);
                    BLower <- c(0,rep(-Inf,length(B)-1));
                    BUpper <- rep(Inf,length(B));
                }
                else{
                    BLower <- rep(-Inf,length(B));
                    BUpper <- rep(Inf,length(B));
                }
            }
            else if(distribution=="dalaplace"){
                if(!aParameterProvided){
                    B <- c(0.5, B);
                    BLower <- c(0,rep(-Inf,length(B)-1));
                    BUpper <- c(1,rep(Inf,length(B)-1));
                }
                else{
                    BLower <- rep(-Inf,length(B));
                    BUpper <- rep(Inf,length(B));
                }
            }
            else if(distribution=="dfnorm"){
                B <- c(sd(y),B);
                BLower <- c(0,rep(-Inf,length(B)-1));
                BUpper <- rep(Inf,length(B));
            }
            else if(any(distribution==c("dgnorm","dlgnorm"))){
                if(!aParameterProvided){
                    B <- c(2,B);
                    BLower <- c(0,rep(-Inf,length(B)-1));
                    BUpper <- rep(Inf,length(B));
                }
                else{
                    BLower <- rep(-Inf,length(B));
                    BUpper <- rep(Inf,length(B));
                }
            }
            else if(distribution=="dbcnorm"){
                if(aParameterProvided){
                    BLower <- rep(-Inf,length(B));
                    BUpper <- rep(Inf,length(B));
                }
                else{
                    BLower <- c(0,rep(-Inf,length(B)-1));
                    BUpper <- c(1,rep(Inf,length(B)-1));
                }
            }
            else{
                BLower <- rep(-Inf,length(B));
                BUpper <- rep(Inf,length(B));
            }
        }

        # Change otU to FALSE everywhere, so that the lags are refitted for the occurrence models
        if(any(distribution==c("plogis","pnorm")) && ariModel){
            otU <- rep(FALSE,obsInsample);
        }

        print_level_hidden <- print_level;
        if(print_level==41){
            print_level[] <- 0;
        }

        if(any(loss==c("LASSO","RIDGE"))){
            denominator <- apply(matrixXreg, 2, sd);
            # No variability, substitute by 1
            denominator[is.infinite(denominator)] <- 1;
            # # If it is lower than 1, then we are probably dealing with (0, 1). No need to normalise
            # denominator[abs(denominator)<1] <- 1;
        }
        else{
            denominator <- NULL;
        }

        # Although this is not needed in case of distribution="dnorm", we do that in a way, for the code consistency purposes
        res <- nloptr(B, CF,
                      opts=list(algorithm=algorithm, xtol_rel=xtol_rel, maxeval=maxeval, print_level=print_level,
                                maxtime=maxtime, xtol_abs=xtol_abs, ftol_rel=ftol_rel, ftol_abs=ftol_abs),
                      lb=BLower, ub=BUpper,
                      distribution=distribution, loss=loss, y=y, matrixXreg=matrixXreg,
                      recursiveModel=recursiveModel, denominator=denominator);
        if(recursiveModel){
            res2 <- nloptr(res$solution, CF,
                           opts=list(algorithm=algorithm, xtol_rel=xtol_rel, maxeval=maxeval, print_level=print_level,
                                maxtime=maxtime, xtol_abs=xtol_abs, ftol_rel=ftol_rel, ftol_abs=ftol_abs),
                           lb=BLower, ub=BUpper,
                           distribution=distribution, loss=loss, y=y, matrixXreg=matrixXreg,
                           recursiveModel=recursiveModel, denominator=denominator);
            if(res2$objective<res$objective){
                res[] <- res2;
            }
        }
        B[] <- res$solution;
        CFValue <- res$objective;

        # A hack for LASSO / RIDGE, lambda==1 and intercept
        if(any(loss==c("LASSO","RIDGE")) && lambda==1 && interceptIsNeeded){
            B[-1] <- 0;
        }

        if(print_level_hidden>0){
            print(res);
        }

        # If there were ARI, write down the polynomial
        if(ariModel){
            # Some models save the first parameter for scale
            nVariablesForReal <- length(B);
            if(all(c(arOrder,iOrder)>0)){
                poly1[-1] <- -B[nVariablesForReal-c(1:arOrder)+1];
                ellipsis$polynomial <- -polyprod(poly2,poly1)[-1];
                ellipsis$arima <- c(arOrder,iOrder,0);
            }
            else if(iOrder>0){
                ellipsis$polynomial <- -poly2[-1];
                ellipsis$arima <- c(0,iOrder,0);
            }
            else{
                ellipsis$polynomial <- B[nVariablesForReal-c(1:arOrder)+1];
                ellipsis$arima <- c(arOrder,0,0);
            }
            names(ellipsis$polynomial) <- ariNames;
            ellipsis$arima <- paste0("ARIMA(",paste0(ellipsis$arima,collapse=","),")");
        }
    }
    # If the parameters are provided
    else{
        if(any(loss==c("LASSO","RIDGE"))){
            denominator <- apply(matrixXreg, 2, sd);
            # No variability, substitute by 1
            denominator[is.infinite(denominator)] <- 1;
            # # If it is lower than 1, then we are probably dealing with (0, 1). No need to normalise
            # denominator[abs(denominator)<1] <- 1;
        }
        else{
            denominator <- NULL;
        }
        # The data are provided, so no need to do recursive fitting
        recursiveModel <- FALSE;
        # If this was ARI, then don't count the AR parameters
        # if(ariModel){
        #     nVariablesExo <- nVariablesExo - ariOrder;
        # }
        B <- parameters;
        nVariables <- length(B);
        if(!is.null(names(B))){
            variablesNames <- names(B);
        }
        else{
            names(B) <- variablesNames;
            names(parameters) <- variablesNames;
        }
        CFValue <- CF(B, distribution, loss, y, matrixXreg, recursiveModel, denominator);
    }

    #### Form the fitted values, location and scale ####
    if(recursiveModel){
        fitterReturn <- fitterRecursive(B, distribution, y, matrixXreg);
        matrixXreg[] <- fitterReturn$matrixXreg;
    }
    else{
        fitterReturn <- fitter(B, distribution, y, matrixXreg);
    }
    mu[] <- fitterReturn$mu;
    scale <- fitterReturn$scale;

    # Give names to additional parameters
    if(is.null(parameters)){
        parameters <- B;
        if(distribution=="dnbinom"){
            if(!aParameterProvided){
                ellipsis$size <- parameters[1];
                parameters <- parameters[-1];
                names(B) <- c("size",variablesNames);
            }
            else{
                names(B) <- variablesNames;
            }
            names(parameters) <- variablesNames;
        }
        else if(distribution=="dchisq"){
            if(!aParameterProvided){
                ellipsis$nu <- nu <- abs(parameters[1]);
                parameters <- parameters[-1];
                names(B) <- c("nu",variablesNames);
            }
            else{
                names(B) <- variablesNames;
            }
            names(parameters) <- c(variablesNames);
        }
        else if(distribution=="dt"){
            if(!aParameterProvided){
                ellipsis$nu <- nu <- abs(parameters[1]);
                parameters <- parameters[-1];
                names(B) <- c("nu",variablesNames);
            }
            else{
                names(B) <- variablesNames;
            }
            names(parameters) <- c(variablesNames);
        }
        else if(distribution=="dfnorm"){
            if(!aParameterProvided){
                ellipsis$sigma <- sigma <- abs(parameters[1]);
                parameters <- parameters[-1];
                names(B) <- c("sigma",variablesNames);
            }
            else{
                names(B) <- variablesNames;
            }
            names(parameters) <- c(variablesNames);
        }
        else if(any(distribution==c("dgnorm","dlgnorm"))){
            if(!aParameterProvided){
                ellipsis$beta <- beta <- abs(parameters[1]);
                parameters <- parameters[-1];
                names(B) <- c("beta",variablesNames);
            }
            else{
                names(B) <- variablesNames;
            }
            names(parameters) <- c(variablesNames);
        }
        else if(distribution=="dalaplace"){
            if(!aParameterProvided){
                ellipsis$alpha <- alpha <- parameters[1];
                parameters <- parameters[-1];
                names(B) <- c("alpha",variablesNames);
            }
            else{
                names(B) <- variablesNames;
            }
            names(parameters) <- variablesNames;
        }
        else if(distribution=="dbeta"){
            if(!FI){
                names(B) <- names(parameters) <- c(paste0("shape1_",variablesNames),paste0("shape2_",variablesNames));
            }
        }
        else if(distribution=="dbcnorm"){
            if(!aParameterProvided){
                ellipsis$lambdaBC <- lambdaBC <- parameters[1];
                parameters <- parameters[-1];
                names(B) <- c("lambda",variablesNames);
            }
            else{
                names(B) <- variablesNames;
            }
            names(parameters) <- variablesNames;
        }
        else{
            names(parameters) <- variablesNames;
            names(B) <- variablesNames;
        }
    }

    if(any(loss==c("LASSO","RIDGE"))){
        ellipsis$lambda <- lambda;
    }

    ### Fitted values in the scale of the original variable
    yFitted[] <- switch(distribution,
                       "dfnorm" = sqrt(2/pi)*scale*exp(-mu^2/(2*scale^2))+mu*(1-2*pnorm(-mu/scale)),
                       "dnorm" =,
                       "dgnorm" =,
                       "dinvgauss" =,
                       "dlaplace" =,
                       "dalaplace" =,
                       "dlogis" =,
                       "dt" =,
                       "ds" =,
                       "dpois" =,
                       "dnbinom" = mu,
                       "dchisq" = mu + nu,
                       "dlnorm" =,
                       "dllaplace" =,
                       "dls" =,
                       "dlgnorm" = exp(mu),
                       "dlogitnorm" = exp(mu)/(1+exp(mu)),
                       "dbcnorm" = bcTransformInv(mu,lambdaBC),
                       "dbeta" = mu / (mu + scale),
                       "pnorm" = pnorm(mu, mean=0, sd=1),
                       "plogis" = plogis(mu, location=0, scale=1)
    );

    ### Error term in the transformed scale
    errors[] <- switch(distribution,
                       "dbeta" = y - yFitted,
                       "dfnorm" =,
                       "dlaplace" =,
                       "dalaplace" =,
                       "dlogis" =,
                       "dt" =,
                       "ds" =,
                       "dnorm" =,
                       "dgnorm" =,
                       "dnbinom" =,
                       "dpois" = y - mu,
                       "dinvgauss" = y / mu,
                       "dchisq" = sqrt(y) - sqrt(mu),
                       "dlnorm" =,
                       "dllaplace" =,
                       "dls" =,
                       "dlgnorm" = log(y) - mu,
                       "dbcnorm" = bcTransform(y,lambdaBC) - mu,
                       "dlogitnorm" = log(y/(1-y)) - mu,
                       "pnorm" = qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1),
                       "plogis" = log((1 + y * (1 + exp(mu))) / (1 + exp(mu) * (2 - y) - y))
                       # Here we use the proxy from Svetunkov & Boylan (2019)
    );

    # If this is the occurrence model, then set unobserved errors to zero
    if(occurrenceModel){
        errors[!otU] <- 0;
    }

    # If negative values are produced in the mu of dbcnorm, correct the fit
    if(distribution=="dbcnorm" && any(mu<0)){
        yFitted[mu<0] <- 0;
    }

    # Parameters of the model + scale
    nParam <- nVariables + (loss=="likelihood")*1;

    # Amend the number of parameters, depending on the type of distribution
    if(any(distribution==c("dnbinom","dchisq","dt","dfnorm","dbcnorm","dgnorm","dlgnorm","dalaplace"))){
        if(!aParameterProvided){
            nParam <- nParam + 1;
        }
    }
    else if(distribution=="dbeta"){
        nParam <- nVariables*2;
        if(!FI){
            nVariables <- nVariables*2;
        }
    }

    # Do not count zero parameters in LASSO / RIDGE
    if(any(loss==c("LASSO","RIDGE"))){
        nParam <- sum(B!=0);
    }

    # If we had huge numbers for cumulative models, fix errors and scale
    if(any(distribution==c("plogis","pnorm")) && (any(is.nan(errors)) || any(is.infinite(errors)))){
        errorsNaN <- is.nan(errors) | is.infinite(errors);

        # Demand occurrs and we predict that
        errors[y==1 & yFitted>0 & errorsNaN] <- 0;
        # Demand occurrs, but we did not predict that
        errors[y==1 & yFitted<0 & errorsNaN] <- 1E+100;
        # Demand does not occurr, and we predict that
        errors[y==0 & yFitted<0 & errorsNaN] <- 0;
        # Demand does not occurr, but we did not predict that
        errors[y==0 & yFitted>0 & errorsNaN] <- -1E+100;

        # Recalculate scale
        scale <- sqrt(meanFast(errors^2));
    }

    #### Produce Fisher Information ####
    if(FI){
        # Only vcov is needed, no point in redoing the occurrenceModel
        occurrenceModel <- FALSE;
        FI <- hessian(CF, B,
                      distribution=distribution, loss=loss, y=y, matrixXreg=matrixXreg,
                      recursiveModel=recursiveModel, denominator=denominator);

        if(any(is.nan(FI))){
            warning("Something went wrong and we failed to produce the covariance matrix of the parameters.\n",
                    "Obviously, it's not our fault. Probably Russians have hacked your computer...\n",
                    "Try a different distribution maybe?", call.=FALSE);
            FI <- diag(1e+100,nVariables);
        }
        dimnames(FI) <- list(variablesNames,variablesNames);
    }

    #### Deal with the occurrence part of the model ####
    if(occurrenceModel){
        mf$subset <- NULL;

        if(interceptIsNeeded){
            # This shit is needed, because R has habit of converting everything into vectors...
            dataWork <- cbind(y,matrixXreg[,-1,drop=FALSE]);
            variablesUsed <- variablesNames[variablesNames!="(Intercept)"];
        }
        else{
            dataWork <- cbind(y,matrixXreg);
            variablesUsed <- variablesNames;
        }
        colnames(dataWork)[1] <- responseName;

        # If there are NaN values, remove the respective observations
        if(any(sapply(mf$data,is.nan))){
            mf$subset <- !apply(sapply(mf$data,is.nan),1,any);
        }
        else{
            mf$subset <- rep(TRUE,obsInsample);
        }

        # New data and new response variable
        dataNew <- dataWork
        # dataNew <- mf$data[mf$subset,,drop=FALSE];
        dataNew[,all.vars(formula)[1]] <- (ot)*1;

        if(!occurrenceProvided){
            occurrence <- do.call("alm", list(formula=formula, data=dataNew, distribution=occurrence, ar=arOrder, i=iOrder));
            if(exists("dataSubstitute",inherits=FALSE,mode="call")){
                occurrence$call$data <- as.name(paste0(deparse(dataSubstitute),collapse=""));
            }
            else{
                occurrence$call$data <- NULL;
            }
        }

        # Corrected fitted (with probabilities)
        yFitted[] <- yFitted * fitted(occurrence);

        # Correction of the likelihood
        CFValue <- CFValue - occurrence$logLik;

        # Change the names of variables used, if ARI was constructed.
        if(ariModel){
            variablesUsed <- variablesUsed[!(variablesUsed %in% arNames)];
            variablesUsed <- c(variablesUsed,ariTransformedNames);
        }
        colnames(dataWork)[1] <- responseName;
        dataWork <- dataWork[,c(responseName, variablesUsed), drop=FALSE]
    }
    else{
        if(interceptIsNeeded){
            # This shit is needed, because R has habit of converting everything into vectors...
            dataWork <- cbind(y,matrixXreg[,-1,drop=FALSE]);
            variablesUsed <- variablesNames[variablesNames!="(Intercept)"];
        }
        else{
            dataWork <- cbind(y,matrixXreg);
            variablesUsed <- variablesNames;
        }
        colnames(dataWork) <- c(responseName, variablesUsed);
    }

    if(distribution=="dbeta"){
        variablesNames <- variablesNames[1:(nVariables/2)];
        nVariables <- nVariables/2;

        # Write down shape parameters
        ellipsis$shape1 <- mu;
        ellipsis$shape2 <- scale;
        # mu and scale are set top be equal to mu and variance.
        scale <- mu * scale / ((mu+scale)^2 * (mu + scale + 1));
        mu <- yFitted;
    }

    # Return LogLik, depending on the used loss
    if(loss=="likelihood"){
        logLik <- -CFValue;
    }
    else if((loss=="MSE" && any(distribution==c("dnorm","dlnorm","dbcnorm","dlogitnorm"))) ||
       (loss=="MAE" && any(distribution==c("dlaplace","dllaplace"))) ||
       (loss=="HAM" && any(distribution==c("ds","dls")))){
        logLik <- -CF(B, distribution, loss="likelihood", y,
                      matrixXreg, recursiveModel, denominator);
    }
    else{
        logLik <- NA;
    }

    finalModel <- structure(list(coefficients=parameters, FI=FI, fitted=yFitted, residuals=as.vector(errors),
                                 mu=mu, scale=scale, distribution=distribution, logLik=logLik,
                                 loss=loss, lossFunction=lossFunction, lossValue=CFValue,
                                 df.residual=obsInsample-nParam, df=nParam, call=cl, rank=nParam,
                                 data=dataWork,
                                 occurrence=occurrence, subset=subset, other=ellipsis, B=B),
                            class=c("alm","greybox"));

    # If this is an occurrence model, flag it as one
    if(any(distribution==c("plogis","pnorm"))){
        class(finalModel) <- c(class(finalModel),"occurrence");
    }

    return(finalModel);
}
