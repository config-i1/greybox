#' Automatic Identification of Demand
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
#' @param ... Other parameters passed to the \code{alm()} function. In case of the
#' \code{aidCat()} function, the \code{aid()} parameters can be passed here.
#'
#' @return \code{aid()} returns an object of class "aid", which contains:
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
#' aid(y)
#'
#' @importFrom stats lowess approx
#' @rdname aid
#' @export
aid <- function(y, ic=c("AICc","AIC","BICc","BIC"), level=0.99,
                loss="likelihood", ...){
    # Intermittent demand identifier

    # Select IC
    ic <- match.arg(ic);
    IC <- switch(ic,"AIC"=AIC,"BIC"=BIC,"BICc"=BICc,AICc);

    obsInSample <- length(y);

    # Substitute NAs with zeroes
    y[is.na(y)] <- 0;

    #### Stockouts detection ####
    # Do stockouts only for data with zeroes
    if(any(y==0)){
        # Demand intervals to identify stockouts/new/old products
        # 0 is for the new products, obsInSample is to track obsolescence
        yIntervals <- diff(c(0,which(y!=0),obsInSample+1));
        xregDataIntervals <- data.frame(y=yIntervals,
                                        ### LOWESS is more conservative than supsmu. Better for identification
                                        # x=supsmu(1:length(yIntervals),yIntervals)$y);
                                        x=lowess(1:length(yIntervals),yIntervals)$y);

        # Apply Geometric distribution model to check for stockouts
        # Use Robust likelihood to get rid of potential strong outliers
        # stockoutModel <-
        # stockoutModelIntercept <-
        #     alm(y-1~1, xregDataIntervals, distribution="dgeom", loss=loss, ...);
        stockoutModel <- alm(y-1~x, xregDataIntervals, distribution="dgeom", loss=loss, ...);

        # if(IC(stockoutModelIntercept)<IC(stockoutModel)){
        #     stockoutModel <- stockoutModelIntercept;
        # }

        probabilities <- pointLikCumulative(stockoutModel);

        # Instead of comparing with the level, see anomalies in comparison with the neighbours!

        # Binaries for new/obsolete products to track zeroes in the beginning/end of data
        productNew <- FALSE;
        productObsolete <- FALSE;

        # If the first one is above the threshold, it is a "new product"
        # if(probabilities[1]>level && yIntervals[1]!=1){
        # If the first one is not one, the data starts with zeroes
        if(yIntervals[1]!=1){
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
    }
    else{
        # message("The data does not contain any zeroes. It must be regular.");
        yIDsToUse <- 1:obsInSample;
        stockoutsStart <- stockoutsEnd <- NULL;
        productNew <- productObsolete <- FALSE;
    }

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

    # Check whether y has only two values
    # The division by max is needed to also check situations with y %in% (0, a), e.g. a=100
    yIsBinary <- all((xregData$y/max(xregData$y)) %in% c(0,1));

    # Or it has just three... very low volume
    yIsLowVolume <- all(xregData$y %in% c(0,1,2));

    # Are there any zeroes left?
    zeroesLeft <- FALSE;
    if(any(y[yIDsToUse]==0)){
        zeroesLeft <- TRUE;
    }

    # Check if the data is integer valued (needed for count data)
    dataIsInteger <- all(y==trunc(y));

    # Type of the data based on the model
    idType <- "regular fractional";

    if(zeroesLeft){
        # Data for demand occurrence
        xregDataOccurrence <- data.frame(y=y[yIDsToUse], x=y[yIDsToUse])
        xregDataOccurrence$y[] <- (xregDataOccurrence$y!=0)*1;
        xregDataOccurrence$x[] <- supsmu(1:length(xregDataOccurrence$y),xregDataOccurrence$y)$y;

        # If there is no variability in SupSmu, use the fixed probability
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

        # List for models
        nModels <- 4
        idModels <- vector("list", nModels);
        names(idModels) <- c("smooth intermittent fractional","lumpy intermittent fractional",
                             "smooth intermittent count","lumpy intermittent count");

        # model 1 is the **smooth intermittent fractional**
        idModels[[1]] <- suppressWarnings(alm(y~., xregData, distribution="drectnorm", loss=loss, ...));

        # This is the basic model
        idType[] <- "smooth intermittent fractional";

        # model 2 is the **lumpy intermittent fractional** (mixture model)
        # model 3 is the **smooth intermittent count** (binary model)
        # If the data is just binary, don't do the mixture model, do the Bernoulli
        if(yIsBinary){
            idModels[[3]] <- modelOccurrence;

            idType[] <- "smooth intermittent count";
        }
        else{
            idModels[[2]] <- suppressWarnings(alm(y~., xregData, distribution="dnorm",
                                                  occurrence=modelOccurrence, loss=loss, ...));
        }

        # This is the lumpy intermittent
        # if(!yIsBinary && (IC(idModels[[2]]) < IC(idModels[[1]]))){
        #     idType[] <- "lumpy intermittent fractional";
        # }

        if(dataIsInteger && !yIsBinary){
            # model 3 is **smooth intermittent count**: Negative Binomial distribution
            idModels[[3]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom",
                                                  maxeval=500, loss=loss, ...));

            # Several other candidates for the model 3
            # These models are needed to take care of potential very low volume data
            idModelAlternative <- vector("list",2)
            idModelAlternative[[1]] <- alm(y~1, xregData, distribution="dbinom",
                                           loss=loss, ...);
            idModelAlternative[[2]] <- alm(y~., xregData, distribution="dbinom",
                                           loss=loss, ...);
            idModelAlternativeICs <- sapply(idModelAlternative, IC);

            # If it is better, substitute it
            if(any(idModelAlternativeICs < IC(idModels[[3]]))){
                idModels[[3]] <- idModelAlternative[[which.min(idModelAlternativeICs)]];
            }
            #model 4 is **lumpy intermittent count**: Negative Binomial distribution + Bernoulli
            idModels[[4]] <- suppressWarnings(alm(y~., xregData, distribution="dnbinom",
                                                  occurrence=modelOccurrence, maxeval=500, loss=loss, ...));

            # If count is better than the fractional
            # if(IC(idModels[[3]]) < IC(idModels[[1]])){
            #     idType[] <- "smooth intermittent count";
            #
            #     # Is lumpy better than smooth?
            #     if(IC(idModels[[4]]) < IC(idModels[[3]])){
            #         idType[] <- "lumpy intermittent count";
            #     }
            # }

            # Choose the best from all the four options
            # idType[] <- names(idModels)[which.min(sapply(idModels, IC))];
        }
    }
    else{
        # List for models
        nModels <- 2
        idModels <- vector("list", nModels);
        names(idModels) <- c("regular fractional", "regular count");

        # model 1 is the **regular fractional**
        idModels[[1]] <- suppressWarnings(alm(y~., xregData,
                                              distribution="dnorm", loss=loss, ...));
        if(dataIsInteger){
            # model 2 is **regular count**: Negative Binomial distribution
            idModels[[2]] <- suppressWarnings(alm(y~., xregData,
                                                  distribution="dnbinom", maxeval=500, loss=loss, ...));
            # if(IC(idModels[[2]]) < IC(idModels[[1]])){
            #     idType[] <- "regular count";
            # }
        }
    }

    # Remove redundant models
    idModels <- idModels[!sapply(idModels, is.null)]
    # Calculate ICs
    # aidCs <- sapply(idModels, IC);
    # Find the best one
    # aidCsBest <- which.min(aidCs);
    if(!yIsBinary){
        # Get its name
        idType <- names(idModels)[which.min(sapply(idModels, IC))];
    }

    # Logical rule: if the demand is integer and intermittent, it is count
    # if((dataIsInteger && idType=="intermittent") ||
    #    # If it is integer-valued with some zeroes, it must be count
    #    (dataIsInteger && zeroesLeft && idType=="regular")){
    #     idType <- "count";
    # }

    # Add stockout model to the output if there were some zeroes
    if(any(y==0)){
        idModels$stockout <- stockoutModel;
    }

    # Stockout dummy variable
    stockoutDummy <- rep(0, obsInSample);
    if(length(stockoutsStart)>0){
        for(i in 1:length(stockoutsStart)){
            stockoutDummy[stockoutsStart[i]:stockoutsEnd[i]] <- 1;
        }
    }
    # Dummy variable for the beginning of series
    newDummy <- rep(0, obsInSample);
    if(productNew){
        newDummy[1:(which(y!=0)[1]-1)] <- 1;
    }
    # Dummy variable for the end of series
    obsoleteDummy <- rep(0, obsInSample);
    if(productObsolete){
        obsoleteDummy[(tail(which(y!=0),1)+1):obsInSample] <- 1;
    }

    return(structure(list(y=y, models=idModels, type=idType,
                          stockouts=list(start=stockoutsStart, end=stockoutsEnd,
                                         dummy=stockoutDummy, new=newDummy, obsolete=obsoleteDummy),
                          new=productNew, obsolete=productObsolete),
                     class="aid"));
}

#' @export
print.aid <- function(x, ...){
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
plot.aid <- function(x, ...){

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


#' @param data The dataframe or a matrix to which the aid function should be applied
#'
#' @return \code{aidCat()} returns an object of class "aidCat", which contains:
#' \itemize{
#' \item categories - the vector with the names of categories for each time series;
#' \item types - the vector with the count of series in each category;
#' \item anomalies - the vector that contains counts of cases where product was flagged
#' as new, obsolete and having stockouts.
#' }
#' @examples
#' xreg <- cbind(x1=rpois(100,1), x2=rpois(100,2),
#'               x3=rpois(100,5), x4=rnorm(100,10,2))
#' plot(aidCat(xreg))
#' @rdname aid
#' @export
aidCat <- function(data, ...){

    # Apply aid()
    if(is.matrix(data)){
        aidApplied <- apply(data, 2, aid, ...);
    }
    else if(is.data.frame(data) || is.list(data)){
        aidApplied <- lapply(data, aid, ...);
    }

    # Extract categories
    aidCategory <- factor(sapply(aidApplied,"[[","type"),
                          levels=c("regular count","smooth intermittent count","lumpy intermittent count",
                                   "regular fractional","smooth intermittent fractional","lumpy intermittent fractional"));
    aidTable <- table(aidCategory);

    # Count number of new, obsolete and stockouts
    aidNew <- sum(sapply(aidApplied,"[[","new"));
    aidOld <- sum(sapply(aidApplied,"[[","obsolete"));
    aidStockouts <- length(unlist(sapply(lapply(aidApplied,"[[","stockouts"),"[[","start")));

    return(structure(list(categories=aidCategory,
                          types=matrix(aidTable,2,3,
                                       dimnames=list(c("Count","Fractional"),
                                                     c("Regular","Smooth Intermittent","Lumpy Intermittent")),
                                       byrow=TRUE),
                          anomalies=setNames(c(aidNew,aidStockouts,aidOld),
                                             c("New","Stockouts","Old"))),
                     class="aidCat"))
}


#' @export
print.aidCat <- function(x, ...){
    cat("Demand categories:\n")
    print(x$types)

    cat("\nAnomalies:\n")
    print(x$anomalies)
}

#' @export
plot.aidCat <- function(x, ...){
    parDefault <- par(mar=c(1,0,1,0));
    on.exit(par(parDefault));

    # The canvas
    plot(0, 0, type="l", xlim=c(-1,2.2), ylim=c(-1.2,1), axes=F, xlab="", ylab="", ...)
    title(xlab="Demand intervals", line=-0.5)
    title(ylab="Demand sizes type", line=-1)
    # abline(h=0)
    # abline(v=c(-1,0,1,2))
    lines(c(-1,2.2),c(-1,-1))
    lines(c(-1,2.2),c(0,0))
    lines(c(-1,2.2),c(1,1))

    lines(c(-1,-1),c(-1.2,1))
    lines(c(0,0),c(-1.2,1))
    lines(c(1,1),c(-1.2,1))
    lines(c(2,2),c(-1.2,1))

    # Categories
    text(-0.5,0.5,paste0("Regular Count\n(",x$types[1,1],")"))
    text(0.5,0.5,paste0("Smooth Intermittent Count\n(",x$types[1,2],")"))
    text(1.5,0.5,paste0("Lumpy Intermittent Count\n(",x$types[1,3],")"))
    text(-0.5,-0.5, paste0("Regular Fractional\n(",x$types[2,1],")"))
    text(0.5,-0.5,paste0("Smooth Intermittent Fractional\n(",x$types[2,2],")"))
    text(1.5,-0.5,paste0("Lumpy Intermittent Fractional\n(",x$types[2,3],")"))

    # Summary
    text(-0.5,-1.1,paste0("Regular\n(",sum(x$types[,1]),")"))
    text(0.5,-1.1,paste0("Smooth Intermittent\n(",sum(x$types[,2]),")"))
    text(1.5,-1.1,paste0("Lumpy Intermittent\n(",sum(x$types[,3]),")"))

    text(2.1,0.5,paste0("Count\n(",sum(x$types[1,]),")"), srt=90)
    text(2.1,-0.5,paste0("Fractional\n(",sum(x$types[2,]),")"), srt=90)

    text(2.1,-1.1,paste0("Total\n(",sum(x$types),")"))

}
