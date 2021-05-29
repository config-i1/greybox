
#### Coefficients and extraction functions ####
#' Scale Model
#'
#' This method produces a model for scale of distribution for the provided pre-estimated model.
#' The model can be estimated either via \code{lm} or \code{alm}.
#'
#' This function is useful, when you suspect a heteroscedasticity in your model and want to
#' fit a model for the scale of the pre-specified distribution. This function is complementary
#' for \code{lm} or \code{alm}.
#'
#' @template author
#'
#' @param model The pre-estimated \code{alm} or \code{lm} model.
#' @param formula The formula for scale. It should start with ~ and contain all variables
#' that should impact the scale.
#' @param data The data, on which the scale model needs to be estimated. If not provided,
#' then the one used in the \code{model} is used.
#' @param parameters The parameters to use in the model. Only needed if you know the parameters
#' in advance or want to test yours.
#' @param ... Other parameters to pass to the method, including those explained in
#' \link[greybox]{alm}.
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' ourModel <- alm(y~.,xreg)
#' ourScale <- sm(ourModel,formula=~x1+x2)
#'
#' @rdname sm
#' @export
sm <- function(model, formula=NULL, data=NULL, parameters=NULL, ...) UseMethod("sm")

#' @rdname sm
#' @export
sm.default <- function(model, formula=NULL, data=NULL, parameters=NULL, ...){
    stop("Sorry, the general method for scale is not supported");
}

#' @rdname sm
#' @export
sm.lm <- function(model, formula=NULL, data=NULL,
                  parameters=NULL, ...){
    # The function creates a scale model for the provided model
    distribution <- "dnorm";
    cl <- model$call;
    if(is.null(data)){
        data <- model$model;
    }
    if(is.null(formula)){
        formula <- formula(model);
        formula[[2]] <- NULL;
    }
    return(do.call("scaler",list(formula, data, model$call$subset, model$call$na.action,
                                 distribution, fitted(model), actuals(model), residuals(model),
                                 parameters, cl=cl, ...)));
}

#' @rdname sm
#' @export
sm.alm <- function(model, formula=NULL, data=NULL,
                   # orders=c(0,0,0),
                   parameters=NULL, ...){
    # The function creates a scale model for the provided model
    distribution <- model$distribution;
    cl <- model$call;
    if(is.null(data)){
        data <- model$call$data;
    }
    if(is.null(formula)){
        formula <- formula(model);
        formula[[2]] <- NULL;
    }
    return(do.call("scaler",list(formula, data, model$call$subset, model$call$na.action,
                                 distribution, fitted(model), actuals(model), residuals(model),
                                 parameters, model$occurrence, model$other, cl=cl, ...)));
}

scaler <- function(formula, data, subset=NULL, na.action=NULL, distribution, mu, y, residuals,
                   parameters=NULL, occurrence=NULL, other=NULL, ...){
    # The function estimates the scale model

    # Start measuring the time of calculations
    startTime <- Sys.time();

    obsInsample <- length(y);

    #### Ellipsis values ####
    ellipsis <- list(...);
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
        ftol_rel <- 1E-6;
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
    print_level_hidden <- print_level;
    if(print_level==41){
        print_level[] <- 0;
    }
    cl <- match.call();
    if(is.null(ellipsis$cl)){
        # This is needed in order to have a reasonable formula saved, so that there are no issues with it
        cl$formula <- eval(cl$formula);
    }
    else{
        cl$data <- quote(ellipsis$cl$data);
    }
    if(is.null(ellipsis$stepSize)){
        stepSize <- .Machine$double.eps^(1/4);
    }
    else{
        stepSize <- ellipsis$stepSize;
    }

    occurrenceModel <- FALSE;
    obsZero <- 0;
    if(is.occurrence(occurrence)){
        occurrenceModel[] <- TRUE;
        obsZero[] <- sum(actuals(occurrence)==0);
    }

    # Prepare the data
    mf <- match.call(expand.dots = FALSE);
    m <- match(c("formula", "data", "subset", "na.action"), names(mf), 0L);
    mf <- mf[c(1L, m)];
    mf$drop.unused.levels <- TRUE;
    if(is.name(mf$data)){
        mf$data <- eval(mf$data);
    }
    if(!is.data.frame(mf$data)){
        mf$data <- data.frame(mf$data);
    }
    mf[[1L]] <- quote(stats::model.frame);
    dataWork <- eval(mf, parent.frame());
    dataTerms <- terms(dataWork);
    matrixXregScale <- model.matrix(dataWork,data=dataWork);
    # Extract number of variables and their names
    nVariables <- ncol(matrixXregScale);
    variablesNames <- colnames(matrixXregScale);

    # Prepare parameters
    if(is.null(B)){
        if(any(distribution==c("dnorm","dlnorm","dbcnorm","dlogitnorm","dfnorm","dlogis"))){
            B <- .lm.fit(matrixXregScale,2*log(abs(residuals)))$coefficients;
        }
        else if(any(distribution==c("dlaplace","dllaplace","dalaplace"))){
            B <- .lm.fit(matrixXregScale,log(abs(residuals)))$coefficients;
        }
        else if(any(distribution==c("ds","dls"))){
            B <- .lm.fit(matrixXregScale,0.5*log(abs(residuals)))$coefficients;
        }
        else if(any(distribution==c("dgnorm","dlgnorm"))){
            B <- .lm.fit(matrixXregScale,other+other*log(abs(residuals)))$coefficients;
        }
        else if(distribution=="dgamma"){
            B <- .lm.fit(matrixXregScale,2*log(abs(residuals-1)))$coefficients;
        }
        else if(distribution=="dinvgauss"){
            B <- .lm.fit(matrixXregScale,log(abs(residuals-1)^2/residuals))$coefficients;
        }
        # Other distributions: dt, dchisq, dnbinom, dpois, pnorm, plogis, dbeta
        else{
            B <- .lm.fit(matrixXregScale,log(abs(residuals)))$coefficients;
        }
    }
    names(B) <- variablesNames;

    BLower <- rep(-Inf,nVariables);
    BUpper <- rep(Inf,nVariables);

    #### Define what to do with the maxeval ####
    if(is.null(ellipsis$maxeval)){
        maxeval <- length(B) * 40;
    }
    else{
        maxeval <- ellipsis$maxeval;
    }

    fitter <- function(B, distribution){
        scale <- exp(matrixXregScale %*% B);
        scale[] <- switch(distribution,
                          "dnorm"=,
                          "dlnorm"=,
                          "dbcnorm"=,
                          "dlogitnorm"=,
                          "dfnorm"=,
                          "dlogis"=sqrt(scale),
                          "dlaplace"=,
                          "dllaplace"=,
                          "dalaplace"=scale,
                          "ds"=,
                          "dls"=scale^2,
                          "dgnorm"=,
                          "dlgnorm"=scale^{1/other},
                          "dgamma"=sqrt(scale)+1,
                          # This is based on polynomial from y = (x-1)^2/x
                          "dinvgauss"=(scale+2+sqrt(scale^2+4*scale))/2,
                          scale);
        return(scale);
    }

    #### The function estimates parameters of scale model ####
    CF <- function(B){
        scale <- fitter(B, distribution);
        CFValue <- -sum(switch(distribution,
                               "dnorm" = dnorm(y, mean=mu, sd=scale, log=TRUE),
                               "dlaplace" = dlaplace(y, mu=mu, scale=scale, log=TRUE),
                               "ds" = ds(y, mu=mu, scale=scale, log=TRUE),
                               "dgnorm" = dgnorm(y, mu=mu, scale=scale,
                                                 shape=other, log=TRUE),
                               "dlogis" = dlogis(y, location=mu, scale=scale, log=TRUE),
                               "dt" = dt(y-mu, df=scale, log=TRUE),
                               "dalaplace" = dalaplace(y, mu=mu, scale=scale,
                                                       alpha=other, log=TRUE),
                               "dlnorm" = dlnorm(y, meanlog=mu, sdlog=scale, log=TRUE),
                               "dllaplace" = dlaplace(log(y), mu=mu,
                                                      scale=scale, log=TRUE)-log(y),
                               "dls" = ds(log(y), mu=mu, scale=scale, log=TRUE)-log(y),
                               "dlgnorm" = dgnorm(log(y), mu=mu, scale=scale,
                                                  shape=other, log=TRUE)-log(y),
                               "dbcnorm" = dbcnorm(y, mu=mu, sigma=scale,
                                                   lambda=other, log=TRUE),
                               "dfnorm" = dfnorm(y, mu=mu, sigma=scale, log=TRUE),
                               "dinvgauss" = dinvgauss(y, mean=mu,
                                                       dispersion=scale/mu, log=TRUE),
                               "dgamma" = dgamma(y, shape=1/scale,
                                                 scale=scale*mu, log=TRUE),
                               "dchisq" = dchisq(y, df=scale, ncp=mu, log=TRUE),
                               "dpois" = dpois(y, lambda=mu, log=TRUE),
                               "dnbinom" = dnbinom(y, mu=mu, size=scale, log=TRUE),
                               "dlogitnorm" = dlogitnorm(y, mu=mu, sigma=scale, log=TRUE)
                               # "dbeta" = dbeta(y, shape1=mu, shape2=scale, log=TRUE),
                               # "pnorm" = c(pnorm(mu[ot], mean=0, sd=1, log.p=TRUE),
                               #             pnorm(mu[!ot], mean=0, sd=1, lower.tail=FALSE, log.p=TRUE)),
                               # "plogis" = c(plogis(mu[ot], location=0, scale=1, log.p=TRUE),
                               #              plogis(mu[!ot], location=0, scale=1, lower.tail=FALSE, log.p=TRUE))
        ));

        # The differential entropy for the models with the missing data
        if(occurrenceModel){
            CFValue[] <- CFValue + switch(distribution,
                                          "dnorm" =,
                                          "dfnorm" =,
                                          "dbcnorm" =,
                                          "dlogitnorm" =,
                                          "dlnorm" = obsZero*(log(sqrt(2*pi)*scale)+0.5),
                                          "dgnorm" =,
                                          "dlgnorm" =obsZero*(1/other-
                                                                  log(other /
                                                                          (2*scale*gamma(1/other)))),
                                          # "dinvgauss" = 0.5*(obsZero*(log(pi/2)+1+suppressWarnings(log(scale)))-
                                          #                                 sum(log(mu[!otU]))),
                                          "dinvgauss" = obsZero*(0.5*(log(pi/2)+1+suppressWarnings(log(scale)))),
                                          "dgamma" = obsZero*(1/scale + log(scale) +
                                                                  log(gamma(1/scale)) +
                                                                  (1-1/scale)*digamma(1/scale)),
                                          "dlaplace" =,
                                          "dllaplace" =,
                                          "ds" =,
                                          "dls" = obsZero*(2 + 2*log(2*scale)),
                                          "dalaplace" = obsZero*(1 + log(2*scale)),
                                          "dlogis" = obsZero*2,
                                          "dt" = obsZero*((scale+1)/2 *
                                                              (digamma((scale+1)/2)-digamma(scale/2)) +
                                                              log(sqrt(scale) * beta(scale/2,0.5))),
                                          "dchisq" = obsZero*(log(2)*gamma(scale/2)-
                                                                  (1-scale/2)*digamma(scale/2)+
                                                                  scale/2),
                                          # "dbeta" = sum(log(beta(mu[otU],scale[otU]))-
                                          #                   (mu[otU]-1)*
                                          #                   (digamma(mu[otU])-
                                          #                        digamma(mu[otU]+scale[otU]))-
                                          #                   (scale[otU]-1)*
                                          #                   (digamma(scale[otU])-
                                          #                        digamma(mu[otU]+scale[otU]))),
                                          # This is a normal approximation of the real entropy
                                          # "dpois" = sum(0.5*log(2*pi*scale)+0.5),
                                          # "dnbinom" = obsZero*(log(sqrt(2*pi)*scale)+0.5),
                                          0
            );
        }
        return(CFValue);
    }

    res <- nloptr(B, CF,
                  opts=list(algorithm=algorithm, xtol_rel=xtol_rel, maxeval=maxeval, print_level=print_level,
                            maxtime=maxtime, xtol_abs=xtol_abs, ftol_rel=ftol_rel, ftol_abs=ftol_abs),
                  lb=BLower, ub=BUpper);

    B[] <- res$solution;
    CFValue <- res$objective;

    if(print_level_hidden>0){
        print(res);
    }

    scale <- fitter(B, distribution);
    #### !!!! This needs to be double checked
    errors <- switch(distribution,
                     "dnorm"=,
                     "dlnorm"=,
                     "dbcnorm"=,
                     "dlogitnorm"=,
                     "dfnorm"=,
                     "dlogis"=,
                     "dlaplace"=,
                     "dllaplace"=,
                     "dalaplace"=abs(residuals),
                     "ds"=,
                     "dls"=residuals^2,
                     "dgnorm"=,
                     "dlgnorm"=abs(residuals)^{1/other},
                     "dgamma"=abs(residuals),
                     "dinvgauss"=abs(residuals))/scale;

    #### Produce Fisher Information ####
    if(FI){
        # Only vcov is needed, no point in redoing the occurrenceModel
        occurrenceModel <- FALSE;
        FI <- hessian(CF, B, h=stepSize);

        if(any(is.nan(FI))){
            warning("Something went wrong and we failed to produce the covariance matrix of the parameters.\n",
                    "Obviously, it's not our fault. Probably Russians have hacked your computer...\n",
                    "Try a different distribution maybe?", call.=FALSE);
            FI <- diag(1e+100,nVariables);
        }
        dimnames(FI) <- list(variablesNames,variablesNames);
    }

    # Form the scale object
    finalModel <- structure(list(formula=formula, coefficients=B, fitted=scale, residuals=errors,
                                 df.residual=obsInsample-nVariables, df=nVariables, call=cl, rank=nVariables,
                                 data=matrixXregScale, terms=dataTerms,
                                 occurrence=occurrence, subset=subset, other=ellipsis, B=B, FI=FI,
                                 timeElapsed=Sys.time()-startTime),
                            class="scale");
    return(finalModel);
}
