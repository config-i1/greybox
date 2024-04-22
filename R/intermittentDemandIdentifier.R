idi <- function(y, ic=c("AIC","AICc","BIC","BICc")){

    # Select IC
    ic <- match.arg(ic);
    IC <- switch(ic,"AIC"=AIC,"BIC"=BIC,"BICc"=BICc,AICc);

    # The original data
    xreg1 <- data.frame(y=y, x=y)
    xreg1$x <- lowess(y)$y;

    # Data for demand sizes
    xreg2Sizes <- data.frame(y=y, x=y)
    xreg2Sizes$x[] <- 0;
    xreg2Sizes$x[y!=0] <- lowess(y[y!=0])$y;

    # Data for demand occurrence
    xreg2Occurrence <- data.frame(y=y, x=y)
    xreg2Occurrence$y[] <- (xreg2Occurrence$y!=0)*1;
    xreg2Occurrence$x[] <- lowess(xreg2Occurrence$y)$y;

    # List for models
    nModels <- 8
    idModels <- vector("list", nModels);

    # model 1 is the normal demand
    idModels[[1]] <- suppressWarnings(alm(y~., xreg1, distribution="dnorm"));

    # model 2 is the normal with occasional stock outs: Normal + Bernoulli(p>0.5)
    idModels[[2]] <- suppressWarnings(alm(y~., xreg1, distribution="dnorm", occurrence="plogis"));

    # model 3 is count data: Negative Binomial distribution
    idModels[[3]] <- suppressWarnings(alm(y~., xreg1, distribution="dnbinom"));

    # model 4 is zero-inflated count data: Negative Binomial distribution + Bernoulli
    modelOccurrence <- suppressWarnings(alm(y~., xreg2Occurrence, distribution="plogis"));
    idModels[[4]] <- suppressWarnings(alm(y~., xreg1, distribution="dnbinom", occurrence=modelOccurrence));

    # model 5 is slow and fractional demand: log-normal/Gamma/IG + Bernoulli
    modelOccurrence <- suppressWarnings(alm(y~., xreg2Occurrence, distribution="plogis"));
    idModels[[5]] <- suppressWarnings(alm(y~., xreg1, distribution="dgamma", occurrence=modelOccurrence));

    # model 6 is lumpy non-seasonal: Normal + Bernoulli(p<0.5)
    modelOccurrence <- suppressWarnings(alm(y~., xreg2Occurrence, distribution="plogis"));
    idModels[[6]] <- suppressWarnings(alm(y~., xreg1, distribution="dnorm", occurrence=modelOccurrence));

    ##### Detect seasonality here #####
    # model 7 is lumpy seasonal: Box-Cox + Bernoulli; pt is seasonal; detect seasonality!
    modelOccurrence <- suppressWarnings(alm(y~., xreg2Occurrence, distribution="plogis"));
    idModels[[7]] <- suppressWarnings(alm(y~., xreg1, distribution="dbcnorm", occurrence=modelOccurrence));

    # model 8 is the demand building up: Box-Cox + Bernoulli; pt increases over time
    modelOccurrence <- suppressWarnings(alm(y~., xreg2Occurrence, distribution="plogis"));
    idModels[[8]] <- suppressWarnings(alm(y~., xreg1, distribution="dbcnorm", occurrence=modelOccurrence));

    # # model 9 is the demand obsolescence: Box-Cox + Bernoulli; pt declines over time
    # modelOccurrence <- suppressWarnings(alm(y~., xreg2Occurrence, distribution="plogis"));
    # idModels[[9]] <- suppressWarnings(alm(y~., xreg1, distribution="dbcnorm", occurrence=modelOccurrence));

    idICs <- sapply(idModels, IC);
    return(structure(list(models=idModels, ICs=idICs),
                     class="idi"))

#     ### This is slow mover with fractional values
#     # First model with rectified normal distribution
#     model1 <- suppressWarnings(alm(y~., xreg1, distribution="drectnorm"));
#
#     ### Erratic demand (low volume with stockouts)
#     # Model for demand occurrence
#     model2Occurrence <- suppressWarnings(alm(y~., xreg2Occurrence, distribution=distribution2Occurrence));
#     # The second model: mixture of Box-Cox Normal and Logistic
#     model2 <- suppressWarnings(alm(y~., xreg2Sizes, distribution=distribution2, occurrence=model2Occurrence));
#
#     ### Regular demand
#     model3 <- suppressWarnings(alm(y~., xreg1, distribution=distribution3));

    # Fix the number of estimated parameters to include occurrence part
    # model2$df <- model2$df + nparam(model2$occurrence)
    # model2$df.residual <- model2$df.residual - nparam(model2$occurrence)

    return(structure(list(intermittent=IC(model2)<IC(model1), model1=model1, model2=model2),
                     class="idi"));
}

# print.idi <- function(x, ...){
#     if(x$intermittent){
#         cat("Data is intermittent\n");
#     }
#     else{
#         cat("Data is not intermittent\n");
#     }
# }
