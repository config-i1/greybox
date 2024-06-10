#' Automatic Demand Identifier
#'
#' The function applies several models on the provided time series and identifies what
#' type of demand it is based on an information criterion.
#'
#' The function creates explanatory variables based on LOWESS of the original data, then
#' applies Normal, Normal + Bernoulli, Negative Binomial and Negative Binomial + Bernoulli
#' models and selects the one that has the lowest IC. Based on that, it decides what type
#' of demand the data corresponds to: regular non-count, intermittent non-count,
#' regular count, or intermittent count.
#'
#' @param y The vector of the data.
#' @param ic Information criterion to use.
#'
#' @return Class "adi" is returned, which contains:
#' \itemize{
#' \item models - All fitted models;
#' \item ICs - Values of information criteria;
#' \item type - The type of the identified demand.
#' }
#'
#' @template author
#' @template keywords
#'
#' @examples
#' # Data from Poisson distribution
#' y <- rpois(120, 0.7)
#' adi(y)
#'
#' @importFrom stats lowess approx
#' @export
adi <- function(y, ic=c("AICc","AIC","BICc","BIC")){
    # Intermittent demand identifier

    # Select IC
    ic <- match.arg(ic);
    IC <- switch(ic,"AIC"=AIC,"BIC"=BIC,"BICc"=BICc,AICc);

    if(all(y!=0)){
        message("The data does not contain any zeroes. It must be regular.");
        return(structure(list(models=NA, ICs=NA, type="regular non-count"),
                     class="adi"))
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

    # Drop x if it does not have much variability (almost constant)
    if(all(round(xregData$x,10)==1)){
        xregData$x <- NULL;
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
    names(idModels) <- c("regular non-count","intermittent non-count","regular count","intermittent count"); #,"intermittent slow");

    # model 1 is the regular demand
    idModels[[1]] <- suppressWarnings(alm(y~., xregData, distribution="dnorm"));
    # idModels[[1]] <- suppressWarnings(stepwise(xregData, distribution="dnorm"));

    # model 2 is the intermittent demand (mixture model)
    idModels[[2]] <- suppressWarnings(alm(y~., xregData, distribution="dnorm", occurrence=modelOccurrence));
    # idModels[[2]] <- suppressWarnings(stepwise(xregData, distribution="dnorm", occurrence=modelOccurrence));

    # If the scale is zero then there must be no variability in demand sizes. Switch them off.
    if(round(idModels[[2]]$scale,10)==0){
        idModels[[2]]$logLik <- idModels[[2]]$occurrence$logLik;
    }

    if(dataIsInteger){
        # model 3 is count data: Negative Binomial distribution
        # idModels[[3]] <- suppressWarnings(alm(y~., xregData, distribution="dpois", maxeval=1000));
        idModels[[3]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom", maxeval=200));
        # idModels[[3]] <- suppressWarnings(stepwise(xregData, distribution="dnbinom"));

        # model 4 is zero-inflated count data: Negative Binomial distribution + Bernoulli
        idModels[[4]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom", occurrence=modelOccurrence, maxeval=200));
        # idModels[[4]] <- suppressWarnings(stepwise(xregData, distribution="dnbinom", occurrence=modelOccurrence));
    }

    # model 5 is slow and fractional demand: Box-Cox Normal + Bernoulli
    # idModels[[5]] <- suppressWarnings(alm(y~., xregData, distribution="dlnorm", occurrence=modelOccurrence));

    # Remove redundant models
    idModels <- idModels[!sapply(idModels, is.null)]
    # Calculate ICs
    adiCs <- sapply(idModels, IC);
    # Find the best one
    adiCsBest <- which.min(adiCs);
    # Get its name
    idType <- names(adiCs)[adiCsBest];

    return(structure(list(models=idModels, ICs=adiCs, type=idType),
                     class="adi"));
}

#' @export
print.adi <- function(x, ...){
    cat("The provided time series is", x$type, "\n");
}
