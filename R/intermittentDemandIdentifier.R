idi <- function(y, ic=c("AIC","AICc","BIC","BICc")){
    # Intermittent demand identifier

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
    nModels <- 5
    idModels <- vector("list", nModels);
    names(idModels) <- c("regular","intermittent","regular count","intermittent count","intermittent slow");

    # model 1 is the regular demand
    idModels[[1]] <- suppressWarnings(alm(y~., xregData, distribution="dnorm"));

    # model 2 is the intermittent demand (mixture model)
    idModels[[2]] <- suppressWarnings(alm(y~., xregData, distribution="dnorm", occurrence=modelOccurrence));

    if(dataIsInteger){
        # model 3 is count data: Negative Binomial distribution
        idModels[[3]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom"));

        # model 4 is zero-inflated count data: Negative Binomial distribution + Bernoulli
        idModels[[4]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom", occurrence=modelOccurrence));
    }

    # model 5 is slow and fractional demand: Box-Cox Normal + Bernoulli
    idModels[[5]] <- suppressWarnings(alm(y~., xregData, distribution="dlnorm", occurrence=modelOccurrence));

    # Remove redundant models
    idModels <- idModels[!sapply(idModels, is.null)]
    # Calculate ICs
    idICs <- sapply(idModels, IC);
    # Find the best one
    idICsBest <- which.min(idICs);
    # Get its name
    idType <- names(idICs)[idICsBest];

    return(structure(list(models=idModels, ICs=idICs, type=idType),
                     class="idi"));
}

print.idi <- function(x, ...){
    cat("The provided time series is", x$type, "\n");
}
