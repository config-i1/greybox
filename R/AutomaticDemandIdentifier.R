#' Automatic Demand Identifier
#'
#' The function applies several models on the provided time series and identifies what
#' type of demand it is based on an information criterion.
#'
#' In the first step, function creates inter-demand intervals and fits a model with LOWESS
#' of it assuming Geometric distribution. The outliers from this model are treated as
#' potential stock outs.
#'
#' In the second step, the function creates explanatory variables based on LOWESS of the
#' original data, then applies Normal, Normal + Bernoulli models and selects the one that
#' has the lowest IC. Based on that, it decides what type of demand the data corresponds
#' to: regular or intermittent. Finally, if the data is count, the function will identify
#' that.
#'
#' @param y The vector of the data.
#' @param ic Information criterion to use.
#' @param level The confidence level used in stockouts identification.
#' @param loss The type of loss function to use in model estimation. See
#' \link[greybox]{alm} for possible options.
#' @param ... Other parameters passed to the \code{alm()} function.
#'
#' @return Class "adi" is returned, which contains:
#' \itemize{
#' \item y - The original data;
#' \item models - All fitted models;
#' \item ICs - Values of information criteria;
#' \item type - The type of the identified demand;
#' \item stockouts - List with start and end ids of potential stockouts;
#' \item new - Binary showing whether the data start with the abnormal number of zeroes.
#' Must be a new product then;
#' \item obsolete - Binary showing whether the data ends with the abnormal number of zeroes.
#' Must be product that was discontinued (obsolete).
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
adi <- function(y, ic=c("AICc","AIC","BICc","BIC"), level=0.99,
                loss="likelihood", ...){
    # Intermittent demand identifier

    # Select IC
    ic <- match.arg(ic);
    IC <- switch(ic,"AIC"=AIC,"BIC"=BIC,"BICc"=BICc,AICc);

    if(all(y!=0)){
        message("The data does not contain any zeroes. It must be regular.");
        return(structure(list(models=NA, ICs=NA, type="regular",
                              new=FALSE, obsolete=FALSE),
                     class="adi"))
    }

    #### Stockouts ####
    # Demand intervals to identify stockouts/new/old products
    # 0 is for the new products, length(y) is to track obsolescence
    yIntervals <- diff(c(0,which(y!=0),length(y)+1));
    xregDataIntervals <- data.frame(y=yIntervals,
                                    x=supsmu(1:length(yIntervals),yIntervals)$y);

    # Apply Geometric distribution model to check for stockouts
    # Use Robust likelihood to get rid of potential strong outliers
    stockoutModel <- alm(y-1~1, xregDataIntervals, distribution="dgeom", loss=loss, ...);
    # stockoutModel <- alm(y-1~x, xregDataIntervals, distribution="dgeom", loss=loss, ...);
    #
    # if(IC(stockoutModelIntercept)<IC(stockoutModel)){
    #     stockoutModel <- stockoutModelIntercept;
    # }

    probabilities <- pointLikCumulative(stockoutModel);

    # Instead of comparing with the level, see anomalies in comparison with the neighbours!

    # Binaries for new/obsolete products to track zeroes in the beginning/end of data
    productNew <- FALSE;
    productObsolete <- FALSE;

    # If the first one is above the threshold, it is a "new product"
    if(probabilities[1]>level && yIntervals[1]!=1){
        productNew[] <- TRUE;
    }
    # If the last one is above the threshold, it must be obsolescence
    if(tail(probabilities,1)>level && tail(yIntervals,1)!=1){
        productObsolete[] <- TRUE;
    }

    # Re-estimate the model dropping those zeroes
    # This is mainly needed to see if LOWESS will be better
    if(any(c(productNew,productObsolete))){
        # Redo indices, dropping the head and the tail if needed
        yIntervalsIDs <- c(c(1)[!productNew],
                           2:(length(yIntervals)-1),
                           length(yIntervals)[!productObsolete]);
        xregDataIntervals <- data.frame(y=yIntervals[yIntervalsIDs]);
                                        # x=supsmu(yIntervalsIDs,yIntervals[yIntervalsIDs])$y);

        # Apply Geometric distribution model to check for stockouts
        stockoutModel <- alm(y-1~1, xregDataIntervals, distribution="dgeom", loss=loss, ...);
        # stockoutModel <- alm(y-1~x, xregDataIntervals, distribution="dgeom", loss=loss, ...);

        # if(IC(stockoutModelIntercept)<IC(stockoutModel)){
        #     stockoutModel <- stockoutModelIntercept;
        # }

        probabilities <- c(c(0)[productNew],
                           pointLikCumulative(stockoutModel),
                           c(0)[productObsolete]);
        # 100 is arbitrary just to have something big, not to flag as outlier
        stockoutModel$mu <- c(c(100)[productNew],
                           stockoutModel$mu,
                           c(100)[productObsolete]);
    }
    else{
        yIntervalsIDs <- 1:length(yIntervals);
    }

    # If the probability is higher than the estimated one, it's not an outlier
    # if(any((probabilities <= 1/stockoutModel$mu) & (probabilities>level))){
    #     probabilities[(probabilities <= 1/stockoutModel$mu) & (probabilities>level)] <- 0;
    # }
    # If the outlier has the interval of 1, it's not an outlier
    if(any(probabilities>level & yIntervals==1)){
        probabilities[probabilities>level & yIntervals==1] <- 0;
    }

    outliers <- probabilities>level;
    outliersID <- which(outliers);

    # Record, when stockouts start and finish
    if(any(outliersID==1)){
        stockoutsStart <- c(1,cumsum(yIntervals)[outliersID-1])+1;
    }
    else{
        stockoutsStart <- cumsum(yIntervals)[outliersID-1]+1;
    }
    stockoutsEnd <- cumsum(yIntervals)[outliersID]-1;

    # IDs of stockouts
    stockoutIDs <- unlist(Map(`:`, stockoutsStart, stockoutsEnd));

    # IDs that will be used in the next models (to drop zeroes in head/tail)
    yIDsFirst <- cumsum(yIntervals)[yIntervalsIDs[1]];
    yIDsLast <- cumsum(yIntervals)[tail(yIntervalsIDs,1)];
    yIDsToUse <- seq(yIDsFirst, yIDsLast, 1)-1;

    # Drop stockout periods
    yIDsToUse <- yIDsToUse[!(yIDsToUse %in% stockoutIDs)];


    #### Checking the demand type ####
    # The original data
    xregData <- data.frame(y=y[yIDsToUse], x=y[yIDsToUse])
    xregData$x <- supsmu(1:length(xregData$y),xregData$y)$y;

    # Data for demand sizes
    # Drop zeroes in the beginning/end based on productNew/productObsolete
    xregDataSizes <- data.frame(y=y[yIDsToUse], x=y[yIDsToUse])
    xregDataSizes$x[] <- 0;
    xregDataSizes$x[xregDataSizes$y!=0] <- supsmu(1:sum(xregDataSizes$y!=0),xregDataSizes$y[xregDataSizes$y!=0])$y;
    # Fill in the gaps for demand sizes
    xregDataSizes$x[xregDataSizes$y==0] <- NA;
    xregDataSizes$x[] <- approx(xregDataSizes$x, xout=c(1:nrow(xregDataSizes)), rule=2)$y;

    # If supsmu didn't work due to high volume of zeroes/ones, use the one from demand sizes
    if(all(xregData$x==0) || all(xregData$x==1)){
        xregData$x[] <- xregDataSizes$x;
    }

    # Drop x if it does not have much variability (almost constant)
    if(all(round(xregData$x,10)==1)){
        xregData$x <- NULL;
    }

    zeroesLeft <- FALSE;
    if(any(y[yIDsToUse]==0)){
        zeroesLeft <- TRUE;
    }

    if(zeroesLeft){
        # Data for demand occurrence
        xregDataOccurrence <- data.frame(y=y[yIDsToUse], x=y[yIDsToUse])
        xregDataOccurrence$y[] <- (xregDataOccurrence$y!=0)*1;
        xregDataOccurrence$x[] <- supsmu(1:length(xregDataOccurrence$y),xregDataOccurrence$y)$y;

        # If there is no variability in LOWESS, use the fixed probability
        if(all(xregDataOccurrence$x==xregDataOccurrence$x[1])){
            modelOccurrence <- suppressWarnings(alm(y~1, xregDataOccurrence, distribution="plogis", loss=loss, ...));
        }
        else{
            # Choose the appropriate occurrence model
            modelOccurrenceFixed <- suppressWarnings(alm(y~1, xregDataOccurrence, distribution="plogis", loss=loss, ...));
            modelOccurrence <- suppressWarnings(alm(y~., xregDataOccurrence, distribution="plogis", loss=loss, ...));

            if(IC(modelOccurrenceFixed)<IC(modelOccurrence)){
                modelOccurrence <- modelOccurrenceFixed;
            }
        }
    }

    # Check if the data is integer valued (needed for count data)
    dataIsInteger <- all(y==trunc(y));

    # List for models
    nModels <- 4
    idModels <- vector("list", nModels);
    names(idModels) <- c("regular","intermittent","count","mixture count");
    #,"intermittent slow");

    # model 1 is the regular demand
    idModels[[1]] <- suppressWarnings(alm(y~., xregData, distribution="dnorm", loss=loss, ...));
    # idModels[[1]] <- suppressWarnings(stepwise(xregData, distribution="dnorm"));

    if(zeroesLeft){
        # If the data is just binary, don't do the mixture model, do the Bernoulli
        # The division by max is needed to also check situations with y %in% (0, a), e.g. a=100
        if(all((xregData$y/max(xregData$y)) %in% c(0,1))){
            idModels[[2]] <- modelOccurrence;
            names(idModels)[2] <- "binary";
        }
        else{
            # model 2 is the intermittent demand (mixture model)
            idModels[[2]] <- suppressWarnings(alm(y~., xregData, distribution="dlnorm",
                                                  occurrence=modelOccurrence, loss=loss, ...));
            # idModels[[2]] <- suppressWarnings(stepwise(xregData, distribution="dnorm", occurrence=modelOccurrence));
        }
    }

    if(dataIsInteger){
        # names(idModels) <- c("count","intermittent count");
        # model 3 is count data: Negative Binomial distribution
        # idModels[[3]] <- suppressWarnings(alm(y~., xregData, distribution="dpois"));
        idModels[[3]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom", maxeval=500));
        # idModels[[3]] <- suppressWarnings(stepwise(xregData, distribution="dnbinom"));

        if(zeroesLeft){
            # If the data is just binary then we already have model 2
            if(all((xregData$y/max(xregData$y)) %in% c(0,1))){
                idModels[[4]] <- NULL;
            }
            else{
                # model 4 is zero-inflated count data: Negative Binomial distribution + Bernoulli
                idModels[[4]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom",
                                                      occurrence=modelOccurrence, maxeval=500));
                # idModels[[4]] <- suppressWarnings(stepwise(xregData, distribution="dnbinom", occurrence=modelOccurrence));
            }
        }
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

    # Logical rule: if the demand is integer and intermittent, it is count
    if((dataIsInteger && idType=="intermittent") ||
       # If it is integer-valued with some zeroes, it must be count
       (dataIsInteger && zeroesLeft && idType=="regular")){
        idType <- "count";
    }

    # Add stockout model to the output
    idModels$stockout <- stockoutModel;

    return(structure(list(y=y, models=idModels, ICs=adiCs, type=idType,
                          stockouts=list(start=stockoutsStart, end=stockoutsEnd),
                          new=productNew, obsolete=productObsolete),
                     class="adi"));
}

#' @export
print.adi <- function(x, ...){
    if(length(x$stockouts$start)>1){
        cat("There are",length(x$stockouts$start),"potential stockouts in the data.\n");
    }
    else if(length(x$stockouts$start)>0){
        cat("There is 1 potential stockout in the data\n");
    }
    if(x$new){
        cat("The product is new (sales start later)\n");
    }
    if(x$obsolete){
        cat("The product has become obsolete\n");
    }
    cat("The provided time series is", x$type, "\n");
}

#' @export
plot.adi <- function(x, ...){

    ids <- time(x$y);
    plot(x$y, ...);
    if(length(x$stockouts$start)>0){
        # Put lines just a bit before and after the stockouts
        idsStockoutsStart <- ids[x$stockouts$start-1]+0.5*difftime(ids[x$stockouts$start],ids[x$stockouts$start-1]);
        idsStockoutsEnd <- ids[x$stockouts$end+1]-0.5*difftime(ids[x$stockouts$end+1],ids[x$stockouts$end]);
        # Plot the lines
        abline(v=idsStockoutsStart, col=2);
        abline(v=idsStockoutsEnd, col=3, lty=2);
        # Get the ylim used for plotting
        ylim <- par("yaxp")[1:2];
        # Add the grey areas
        rect(idsStockoutsStart, ylim[1]-max(x$y), idsStockoutsEnd, ylim[2]+max(x$y),
             col="lightgrey", border=NA, density=20)
    }
}
