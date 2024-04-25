idi <- function(y, ic=c("AIC","AICc","BIC","BICc")){

    # Select IC
    ic <- match.arg(ic);
    IC <- switch(ic,"AIC"=AIC,"BIC"=BIC,"BICc"=BICc,AICc);

    if(all(y!=0)){
        message("The data does not contain any zeroes. It must be regular.",
                call.=FALSE);
        return(structure(list(models=NA, ICs=NA, type="regular"),
                     class="idi"))
    }

    # The original data
    xregData <- data.frame(y=y, x=y)
    xregData$x <- lowess(y)$y;

    # Data for demand sizes
    xregDataSizes <- data.frame(y=y, x=y)
    xregDataSizes$x[] <- 0;
    xregDataSizes$x[y!=0] <- lowess(y[y!=0])$y;
    # Fill in the gaps for demand sizes
    xregDataSizes$x[y==0] <- NA;
    xregDataSizes$x[] <- approx(xregDataSizes$x, xout=c(1:length(y)), rule=2)$y;

    # If LOWESS didn't work due to high volume of zeroes/ones, use the one from demand sizes
    if(all(xregData$x==0) || all(xregData$x==1)){
        xregData$x[] <- xregDataSizes$x;
    }

    # Data for demand occurrence
    xregDataOccurrence <- data.frame(y=y, x=y)
    xregDataOccurrence$y[] <- (xregDataOccurrence$y!=0)*1;
    xregDataOccurrence$x[] <- lowess(xregDataOccurrence$y)$y;

    # If there is no variability in LOWESS, use the fixed probability
    if(all(xregDataOccurrence$x==xregDataOccurrence$x[1])){
        modelOccurrence <- suppressWarnings(alm(y~1, xregDataOccurrence, distribution="plogis"));
    }
    else{
        # Choose the appropriate occurrence model
        modelOccurrenceFixed <- suppressWarnings(alm(y~1, xregDataOccurrence, distribution="plogis"));
        modelOccurrence <- suppressWarnings(alm(y~., xregDataOccurrence, distribution="plogis"));

        if(IC(modelOccurrenceFixed)<IC(modelOccurrence)){
            modelOccurrence <- modelOccurrenceFixed;
        }
    }

    # Check if the data is integer valued (needed for count data)
    dataIsInteger <- all(y==trunc(y));

    # List for models
    nModels <- 4
    idModels <- vector("list", nModels);
    idICs <- vector("numeric", nModels);
    idICs[] <- Inf;

    # Types of data
    idType <- "regular";
    idSubtype <- "none";

    # model 1 is the regular demand
    idModels[[1]] <- suppressWarnings(alm(y~., xregData, distribution="dnorm"));

    # model 2 is the intermittent demand (mixture model)
    idModels[[2]] <- suppressWarnings(alm(y~., xregData, distribution="dnorm", occurrence=modelOccurrence));

    names(idModels)[1:2] <- c("regular","intermittent");

    idICs[1:2] <- sapply(idModels[1:2], IC);

    #### Regular demand ####
    # If model 1 is better, we have regular demand
    if(IC(idModels[[1]]) < IC(idModels[[2]])){

        names(idModels)[3] <- c("regular count");
        if(dataIsInteger){
            # model 3 is count data: Negative Binomial distribution
            idModels[[3]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom"));

            idICs[3] <- IC(idModels[[3]]);

            if(idICs[3]<idICs[1]){
                idSubtype[] <- "count";
            }

            # model 4 is zero-inflated count data: Negative Binomial distribution + Bernoulli
            idModels[[4]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom", occurrence=modelOccurrence));
            names(idModels)[4] <- c("intermittent count");
            idICs[4] <- IC(idModels[[4]]);
            if(idICs[4]<min(idICs[1:3])){
                idType[] <- "intermittent";
            }
        }
        else{
            idICs <- idICs[1:3];
            idModels[[4]] <- NULL;
        }
    }
    #### Intermittent demand is here ####
    else{
        idType[] <- "intermittent";
        idSubtype[] <- "lumpy";

        names(idModels)[3] <- c("intermittent count");

        if(dataIsInteger){
            # model 3 is zero-inflated count data: Negative Binomial distribution + Bernoulli
            idModels[[3]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom", occurrence=modelOccurrence));


            idICs[3] <- IC(idModels[[3]]);
            if(idICs[3]<idICs[2]){
                idSubtype[] <- "count";
            }
        }

        # model 5 is slow and fractional demand: Box-Cox Normal + Bernoulli
        idModels[[4]] <- suppressWarnings(alm(y~., xregData, distribution="dlnorm", occurrence=modelOccurrence));

        names(idModels)[4] <- "intermittent slow";

        idICs[4] <- IC(idModels[[4]]);

        if(idICs[4]<idICs[2]){
            idSubtype[] <- "slow";
        }
    }

    names(idICs) <- names(idModels);

    return(structure(list(models=idModels, ICs=idICs, type=idType,
                          subtype=idSubtype),
                     class="idi"));


    # # model 3 is count data: Negative Binomial distribution
    # idModels[[3]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom"));
    #
    # # model 4 is zero-inflated count data: Negative Binomial distribution + Bernoulli
    # idModels[[4]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom", occurrence=modelOccurrence));
    #
    # # model 5 is slow and fractional demand: Box-Cox Normal + Bernoulli
    # modelOccurrence <- suppressWarnings(alm(y~., xregDataOccurrence, distribution="plogis"));
    # idModels[[5]] <- suppressWarnings(alm(y~., xregData, distribution="dbcnorm", occurrence=modelOccurrence));
    #
    # # model 6 is lumpy non-seasonal: Normal + Bernoulli(p<0.5)
    # modelOccurrence <- suppressWarnings(alm(y~., xregDataOccurrence, distribution="plogis"));
    # idModels[[6]] <- suppressWarnings(alm(y~., xregData, distribution="dnorm", occurrence=modelOccurrence));
    #
    # # model 7 is lumpy seasonal: Box-Cox + Bernoulli; pt is seasonal; detect seasonality!
    # # modelOccurrence <- suppressWarnings(alm(y~., xregDataOccurrence, distribution="plogis"));
    # # idModels[[7]] <- suppressWarnings(alm(y~., xregData, distribution="dbcnorm", occurrence=modelOccurrence));
    #
    # # model 8 is the demand building up: Box-Cox + Bernoulli; pt increases over time
    # # modelOccurrence <- suppressWarnings(alm(y~., xregDataOccurrence, distribution="plogis"));
    # # idModels[[8]] <- suppressWarnings(alm(y~., xregData, distribution="dbcnorm", occurrence=modelOccurrence));
    #
    # # # model 9 is the demand obsolescence: Box-Cox + Bernoulli; pt declines over time
    # # modelOccurrence <- suppressWarnings(alm(y~., xregDataOccurrence, distribution="plogis"));
    # # idModels[[9]] <- suppressWarnings(alm(y~., xregData, distribution="dbcnorm", occurrence=modelOccurrence));
    #
    # idICs <- sapply(idModels, IC);
    # idICBest <- which.min(idICs)[1];
    #
    # idType <- switch(idICBest,
    #                  "1"="regular",
    #                  "2"="erratic",
    #                  "3"="count",
    #                  "4"="count zi",
    #                  "5"="slow & fractional",
    #                  "6"="lumpy");
    #
    # idSubtype <- "none";
    # # if(idType=="slow & fractional"){
    # #     if(xregDataOccurrence$x)
    # #     idSubtype <- fitted(idModels[[idICBest]]$occurrence);
    # # }
    #
    # return(structure(list(models=idModels, ICs=idICs, type=idType,
    #                       subtype=idSubtype),
    #                  class="idi"));

    # ### This is slow mover with fractional values
    # # First model with rectified normal distribution
    # model1 <- suppressWarnings(alm(y~., xregData, distribution="drectnorm"));
    #
    # ### Erratic demand (low volume with stockouts)
    # # Model for demand occurrence
    # model2Occurrence <- suppressWarnings(alm(y~., xregDataOccurrence, distribution=distribution2Occurrence));
    # # The second model: mixture of Box-Cox Normal and Logistic
    # model2 <- suppressWarnings(alm(y~., xregDataSizes, distribution=distribution2, occurrence=model2Occurrence));
    #
    # ### Regular demand
    # model3 <- suppressWarnings(alm(y~., xregData, distribution=distribution3));

    # Fix the number of estimated parameters to include occurrence part
    # model2$df <- model2$df + nparam(model2$occurrence)
    # model2$df.residual <- model2$df.residual - nparam(model2$occurrence)

    # return(structure(list(intermittent=IC(model2)<IC(model1), model1=model1, model2=model2),
    #                  class="idi"));
}

print.idi <- function(x, ...){
    # print(x$ICs);
    cat(x$type, x$subtype);
}
