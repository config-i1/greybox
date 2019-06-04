#' Regression for Multiple Comparison
#'
#' RMC stands for "Regression for Multiple Comparison", referring to the
#' comparison of forecasting methods. This is a parametric test for the comparison
#' of means of several distributions
#
#' This test is a parametric counterpart of Nemenyi / MCB test (Demsar, 2006) and
#' uses asymptotic properties of regression models. It relies on distributional
#' assumptions about the provided data. For instance, if the mean forecast errors
#' are used, then it is safe to assume that the regression model constructed on
#' them will have symmetrically distributed residuals, thus normal regression can
#' be used for the parameters estimation.
#'
#' The test constructs the regression model of the type:
#'
#' y = b' X + e,
#'
#' where y is the vector of the provided data (as.vector(data)), X is the matrix
#' of dummy variables for each column of the data (forecasting method), b is the
#' vector of coefficients for the dummies and e is the error term of the model.
#'
#' Depending on the provided data, it might make sense to use different types of
#' regressions. The default one is the log normal distribution for the relative
#' error measures. The type of distribution is regulated with \code{distribution}
#' and is restricted by the values of it from the \link[greybox]{alm} function.
#'
#' The advisable error measures to use in the test are relative measures, such as
#' RelMAE, RelRMSE, RelAME. They are unbiased and their logarithms are symmetrically
#' distributed (Davydenko & Fildes, 2013). Although their distributions are not log
#' normal, given the typically large samples of datasets, the Central Limit
#' Theorem helps in the adequate construction of the confidence intervals for the
#' parameters.
#'
#' The test is equivalent to Nemenyi test, when applied to the ranks of the error
#' measures on large samples with \code{distribution="dnorm"}.
#'
#' There is also a \code{plot()} method that allows producing either "mcb" or "lines"
#' style of plot. This can be regulated via \code{plot(x, outplot="lines")}.
#
#' @param data Matrix or data frame with observations in rows and variables in
#' columns.
#' @param distribution Type of the distribution to use. This value is passed to
#' \code{alm()} function. \code{"dlnorm"} would lead to the alm with log normal
#' distribution. If this is a clear forecast error, then \code{"dnorm"} might be
#' appropriate, leading to a simple Gausian linear regression. \code{"dfnorm"}
#' would lead to an alm model with folded normal distribution. You can try some
#' other distributions, but don't expect anything meaningful.
#' @param level The width of the confidence interval. Default is 0.95.
#' @param outplot What type of plot to use after the calculations. This can be
#' either "MCB" (\code{"mcb"}), or "Vertical lines" (\code{"lines"}), or nothing
#' (\code{"none"}). You can also use plot method on the produced object in order
#' to get the same effect.
#' @param select What column of data to highlight on the plot. If NULL, then
#' the method with the lowest value is selected.
#' @param ... Other parameters passed to plot function
#
#' @return If \code{outplot!="none"}, then the function plots the results after all
#' the calculations. In case of \code{distribution="dnorm"}, the closer to zero the
#' intervals are, the better model performs. When \code{distribution="dlnorm"} or
#' \code{distribution="dfnorm"}, the lower, the better.
#'
#' Function returns a list of a class "rmc", which contains the following
#' variables:
#' \itemize{
#' \item{mean}{Mean values for each method.}
#' \item{interval}{Confidence intervals for each method.}
#' \item{vlines}{Coordinates used for outplot="l", marking the groups of methods.}
#' \item{groups}{The table containing the groups. \code{TRUE} - methods are in the
#' same group, \code{FALSE} - they are not.}
#' \item{methods}{Similar to \code{group} parameter, but with a slightly different
#' presentation.}
#' \item{p.value}{p-value for the test of the significance of the model. This is a
#' log-likelihood ratios chi-squared test, comparing the model with the one with
#' intercept only.}
#' \item{importance}{The weights of the estimated model in comparison with the
#' model with the constant only. 0 means that the constant is better, 1 means that
#' the estimated model is the best.}
#' \item{level}{Significance level.}
#' \item{model}{lm model produced for the calculation of the intervals.}
#' \item{outplot}{Style of the plot to produce.}
#' \item{select}{The selected variable to highlight.}
#' }
#
#' @keywords htest
#' @template author
#
#' @references \itemize{
#' \item  Demsar, J. (2006). Statistical Comparisons of Classifiers over
#' Multiple Data Sets. Journal of Machine Learning Research, 7, 1-30.
#' \url{http://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf}
#' \item Davydenko, A., Fildes, R. (2013). Measuring Forecasting Accuracy:
#' The Case Of Judgmental Adjustments To Sku-Level Demand Forecasts.
#' International Journal of Forecasting, 29(3), 510-522.
#' \url{https://doi.org/10.1016/j.ijforecast.2012.09.002}
#' \item Hea-Jung Kim (2006) On the Ratio of Two Folded Normal
#' Distributions, Communications in Statistics Theory and Methods, 35:6,
#' 965-977, \url{https://doi.org/10.1080/03610920600672229}
#' }
#
#' @seealso \code{\link[greybox]{alm}}
#'
#' @examples
#
#' N <- 50
#' M <- 4
#' ourData <- matrix(rnorm(N*M,mean=0,sd=1), N, M)
#' ourData[,2] <- ourData[,2]+4
#' ourData[,3] <- ourData[,3]+3
#' ourData[,4] <- ourData[,4]+2
#' colnames(ourData) <- c("Method A","Method B","Method C - long name","Method D")
#' rmc(ourData, distribution="dnorm", level=0.95)
#
#' # In case of AE-based measures, distribution="dfnorm" should be selected
#' rmc(abs(ourData), distribution="dfnorm", level=0.95)
#'
#' # APE-based measures should not be used in general...
#'
#' # If RelMAE or RelMSE is used for measuring data, then it makes sense to use
#' # distribution="dlnorm" for the RelMAE / RelMSE, as it can be approximated by
#' # log normal distribution, because according to Davydenko & Fildes (2013) the
#' # logarithms of these measures have symmetric distribution.
#' ourTest <- rmc((abs(ourData) / rfnorm(N, 0.3, 1)), distribution="dlnorm", level=0.95)
#' # The exponents of mean values from this function will correspond to the
#' # geometric means of RelMAE / RelMSE.
#' exp(ourTest$mean)
#' # The same is for the intervals:
#' exp(ourTest$interval)
#'
#' # You can also reproduce plots in different styles:
#' plot(ourTest, outplot="lines")
#'
#' # Or you can use the default "mcb" style and set additional parameters for the plot():
#' par(mar=c(2,2,4,0)+0.1)
#' plot(ourTest, main="Four methods")
#'
#' # The following example should give similar results to Nemenyi test on
#' # large samples, which compares medians of the distributions:
#' rmc(t(apply(ourData,1,rank)), distribution="dnorm", level=0.95)
#'
#' # You can also give a try to SE-based measures with distribution="dchisq":
#' rmc(ourData^2, distribution="dchisq", level=0.95)
#'
#' @importFrom stats pchisq
#' @export rmc
rmc <- function(data, distribution=c("dlnorm","dnorm","dfnorm"),
                level=0.95, outplot=c("mcb","lines","none"), select=NULL, ...){

    #### This is temporary and needs to be removed at some point! ####
    outplot <- depricator(outplot, list(...));

    distribution <- distribution[1];
    outplot <- match.arg(outplot,c("mcb","lines","none"));

    #### Prepare the data ####
    obs <- nrow(data);
    nMethods <- ncol(data);
    if(nMethods>obs){
        response <- readline(paste0("The number of methods is higher than the number of series. ",
                                    "Are you sure that you want to continue? y/n?"));

        if(all(response!=c("y","Y"))){
            stop(paste0("Number of methods is higher than the number of series. ",
                        "The user aborted the calculations."), call.=FALSE);
        }
    }
    obsAll <- obs*nMethods;
    namesMethods <- colnames(data);

    if(is.null(namesMethods)){
        namesMethods <- paste0("Method",c(1:nMethods));
    }

    # Form the matrix of dummy variables, excluding the first one
    dataNew <- matrix(0,obsAll,nMethods+1);
    for(i in 2:nMethods){
        dataNew[obs*(i-1)+1:obs,i+1] <- 1;
    }
    dataNew[,2] <- 1;

    # Collect stuff to form the matrix
    if(is.data.frame(data)){
        dataNew[,1] <- unlist(data);
    }
    else{
        dataNew[,1] <- c(data);
    }
    # Remove data in order to preserve memory
    rm(data);

    colnames(dataNew) <- c("y","(Intercept)",namesMethods[-1]);

    #### Fit the model ####
    # This is the model used for the confidence intervals calculation
    # And the one needed for the importance and the p-value of the model
    if(distribution=="dnorm"){
        lmModel <- .lm.fit(dataNew[,-1], dataNew[,1]);
        lmModel$xreg <- dataNew[,-1];
        lmModel$df.residual <- obsAll - nMethods;
        class(lmModel) <- c("lmGreybox","lm");

        lmModel$fitted <- dataNew[,1] - residuals(lmModel);

        lmModel2 <- .lm.fit(as.matrix(dataNew[,2]), dataNew[,1]);
        lmModel2$df.residual <- obsAll - 1;
        class(lmModel2) <- c("lmGreybox","lm");
    }
    else if(distribution=="dlnorm"){
        lmModel <- .lm.fit(dataNew[,-1], log(dataNew[,1]));
        lmModel$xreg <- dataNew[,-1];
        lmModel$df.residual <- obsAll - nMethods;
        class(lmModel) <- c("lmGreybox","lm");

        lmModel$fitted <- log(dataNew[,1]) - residuals(lmModel);

        lmModel2 <- .lm.fit(as.matrix(dataNew[,2]), log(dataNew[,1]));
        lmModel2$df.residual <- obsAll - 1;
        class(lmModel2) <- c("lmGreybox","lm");
    }
    else{
        lmModel <- alm(y~., data=dataNew[,-2], distribution=distribution, fast=TRUE);

        lmModel2 <- alm(y~1,data=dataNew[,-2], distribution=distribution, fast=TRUE);
    }

    # Remove dataNew in order to preserve memory
    rm(dataNew);

    #### Extract the parameters ####
    # Construct intervals
    lmCoefs <- coef(lmModel);
    # Force confint to be estimated inside the function
    lmIntervals <- confint(lmModel, level=level)[,-1];
    names(lmCoefs) <- namesMethods;
    rownames(lmIntervals)[1] <- namesMethods[1];
    lmCoefs[-1] <- lmCoefs[1] + lmCoefs[-1];
    lmIntervals[-1,] <- lmCoefs[1] + lmIntervals[-1,];

    #### Relative importance of the model and the test ####
    AICs <- c(AIC(lmModel2),AIC(lmModel));
    delta <- AICs - min(AICs);
    importance <- (exp(-0.5*delta) / sum(exp(-0.5*c(delta))))[2];

    # Chi-squared test. +logLikExp is because the logLik is negative
    p.value <- pchisq(-(logLik(lmModel2)-logLik(lmModel)),
                      lmModel2$df.residual-lmModel$df.residual, lower.tail=FALSE);

    if(is.null(select)){
        select <- which.min(lmCoefs);
    }

    select <- which(namesMethods[order(lmCoefs)]==namesMethods[select]);
    lmIntervals <- lmIntervals[order(lmCoefs),];
    lmCoefs <- lmCoefs[order(lmCoefs)];

    # If this was log Normal, take exponent of the coefficients
    if(distribution=="dlnorm"){
        lmIntervals <- exp(lmIntervals);
        lmCoefs <- exp(lmCoefs);
    }

    # Remove `` symbols in case of spaces in names
    names(lmCoefs) <- gsub("[[:punct:]]", "", names(lmCoefs));
    rownames(lmIntervals) <- names(lmCoefs);


    #### Prepare things for the groups for "lines" plot ####
    # Find groups
    vlines <- matrix(NA, nrow=nMethods, ncol=2);
    for(i in 1:nMethods){
        intersections <- which(!(lmIntervals[,2]<lmIntervals[i,1] | lmIntervals[,1]>lmIntervals[i,2]));
        vlines[i,] <- c(min(intersections),max(intersections));
    }

    # Get rid of duplicates and single member groups
    vlines <- unique(vlines);
    # Re-convert to matrix if necessary and find number of remaining groups
    if(length(vlines)==2){
        vlines <- matrix(vlines,1,2);
    }

    nGroups <- nrow(vlines);
    colnames(vlines) <- c("Group starts","Group ends");
    rownames(vlines) <- paste0("Group",c(1:nGroups));

    groups <- matrix(FALSE, nMethods, nGroups, dimnames=list(names(lmCoefs), rownames(vlines)));
    for(i in 1:nGroups){
        groups[c(vlines[i,1]:vlines[i,2]),i] <- TRUE;
    }

    methodGroups <- matrix(TRUE,nMethods,nMethods,
                           dimnames=list(names(lmCoefs),names(lmCoefs)));
    for(i in 1:nMethods){
        for(j in 1:nMethods){
            methodGroups[i,j] <- ((lmIntervals[i,2] >= lmIntervals[j,1]) & (lmIntervals[i,1] <= lmIntervals[j,1])
                                  | (lmIntervals[i,2] >= lmIntervals[j,2]) & (lmIntervals[i,1] <= lmIntervals[j,2])
                                  | (lmIntervals[i,2] <= lmIntervals[j,2]) & (lmIntervals[i,1] >= lmIntervals[j,1]))
        }
    }

    returnedClass <- structure(list(mean=lmCoefs, interval=lmIntervals, vlines=vlines, groups=groups,
                                    methods=methodGroups, importance=importance, p.value=p.value,
                                    level=level, model=lmModel, outplot=outplot, select=select,
                                    distribution=distribution),
                               class="rmc");
    if(outplot!="none"){
        plot(returnedClass, outplot=outplot, ...);
    }
    else{
        returnedClass$outplot[] <- "mcb";
    }
    return(returnedClass);
}

#' @importFrom graphics axis box
#' @export
plot.rmc <- function(x, outplot=c("mcb","lines"), ...){
    nMethods <- length(x$mean);
    namesMethods <- names(x$mean);

    if(nMethods>5){
        las <- 2;
    }
    else{
        las <- 0;
    }

    args <- list(...);
    argsNames <- names(args);

    if(!("xaxs" %in% argsNames)){
        args$xaxs <- "i";
    }
    if(!("yaxs" %in% argsNames)){
        args$yaxs <- "i";
    }

    args$x <- args$y <- NA;
    args$axes <- FALSE;
    args$pch <- NA;

    if(!("main" %in% argsNames)){
        args$main <- paste0("The p-value from the significance test is ", format(round(x$p.value,3),nsmall=3),".\n",
                         x$level*100,"% confidence intervals constructed.");
    }

    if(ncol(x$groups)>1){
        pointCol <- rep("#0DA0DC", nMethods);
        pointCol[x$methods[,x$select]] <- "#0C6385";
        lineCol <- "#0DA0DC";
    }
    else{
        pointCol <- "darkgrey";
        lineCol <- "grey";
    }

    outplot <- match.arg(outplot,c("mcb","lines"));

    if(outplot=="mcb"){
        if(!("xlab" %in% argsNames)){
            args$xlab <- "";
        }
        if(!("ylab" %in% argsNames)){
            args$ylab <- "";
        }
        # Remaining defaults
        if(!("xlim" %in% argsNames)){
            args$xlim <- c(0,nMethods+1);
        }
        if(!("ylim" %in% argsNames)){
            args$ylim <- range(x$interval);
            args$ylim <- c(args$ylim[1]-0.1,args$ylim[2]+0.1);
        }

        # Use do.call to use manipulated ellipsis (...)
        do.call(plot, args);
        for(i in 1:nMethods){
            lines(rep(i,2),x$interval[i,],col=lineCol,lwd=2);
        }
        points(x$mean, pch=19, col=pointCol);
        axis(1, at=c(1:nMethods), labels=namesMethods, las=las);
        axis(2);
        box(which="plot", col="black");

        abline(h=x$interval[x$select,], lwd=2, lty=2, col="grey");
    }
    else if(outplot=="lines"){
        # Save the current par() values
        parDefault <- par(no.readonly=TRUE);
        parMar <- parDefault$mar;

        vlines <- x$vlines;

        k <- nrow(vlines);
        colours <- c("#0DA0DC","#17850C","#EA3921","#E1C513","#BB6ECE","#5DAD9D");
        colours <- rep(colours,ceiling(k/length(colours)))[1:k];
        groupElements <- apply(x$groups,2,sum);

        labelSize <- max(nchar(namesMethods));

        if(!("ylab" %in% argsNames)){
            args$ylab <- "";
        }
        if(!("xlab" %in% argsNames)){
            args$xlab <- "";
        }
        # Remaining defaults
        if(!("xlim" %in% argsNames)){
            args$xlim <- c(0,k);
        }
        if(!("ylim" %in% argsNames)){
            args$ylim <- c(1,nMethods);
        }

        if(all(parMar==(c(5,4,4,2)+0.1))){
            parMar <- c(2, 2+labelSize/2, 2, 2) + 0.1
        }
        else{
            parMar <- parMar + c(0, labelSize/2, 0, 2)
        }

        if(args$main!=""){
            parMar <- parMar + c(0,0,4,0)
        }
        if(args$ylab!=""){
            parMar <- parMar + c(0,2,0,0);
        }
        if(args$xlab!=""){
            parMar <- parMar + c(2,0,0,0);
        }
        par(mar=parMar);

        # Use do.call to use manipulated ellipsis (...)
        do.call(plot, args);

        if(k>1){
            for(i in 1:k){
                if(groupElements[i]>1){
                    lines(c(i,i), vlines[i,], col=colours[i], lwd = 2);
                    lines(c(0,i), rep(vlines[i,1],times=2), col="gray", lty=2);
                    lines(c(0,i), rep(vlines[i,2],times=2), col="gray", lty=2);
                }
            }
        }
        else{
            lines(c(1,1), c(1,nMethods), col=lineCol, lwd = 2);
            lines(c(0,1), rep(vlines[1,1],times=2), col=lineCol, lty=2);
            lines(c(0,1), rep(vlines[1,2],times=2), col=lineCol, lty=2);
        }
        axis(2, at=c(1:nMethods), labels=namesMethods, las=2);

        par(parDefault)
    }
}

#' @export
print.rmc <- function(x, ...){
    cat(paste0("Regression for Multiple Comparison with ",switch(x$distribution,
                                                                 dnorm="normal",
                                                                 dfnorm="folded normal",
                                                                 dlnorm="log normal",
                                                                 dchisq="Chi-Squared"),
               " distribution.\n"));
    cat(paste0("The siginificance level is ",(1-x$level)*100,"%\n"));
    cat(paste0("The number of observations is ",nobs(x$model),", the number of methods is ",length(x$mean),"\n"));
    cat(paste0("Significance test p-value: ",round(x$p.value,5),"\n"));
}
