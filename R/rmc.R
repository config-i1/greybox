#' RMC test
#'
#' RMC stands for "Regression for Methods Comparison". This is a parametric
#' test for the comparison of means of several distributions
#
#' This test is a parametric counterpart of nemenyi / MCB test (Demsar, 2006) and
#' uses asymptotic properties of regression models. It relies on distributional
#' assumptions about the provided data. For instance, if the mean forecast errors
#' are used, then it is safe to assume that the regression model constructed on
#' them will have normally distributed residuals.
#'
#' The test constructs the regression model of the kind:
#'
#' y = b' X + e,
#'
#' where y is the vector of the provided data (as.vector(data)), X is the matrix
#' of dummy variables for each column of the data (forecasting method), b is the
#' vector of coefficients for the dummies and e is the error term of the model.
#'
#' Depending on the provided data, it might make sense to use different types of
#' regressions. The function supports Gausian linear regression
#' (\code{distribution="dnorm"}, when the data is normal), advanced linear regression with
#' folded normal distribution (\code{distribution="dfnorm"}, for example, absolute errors,
#' assuming that the original errors are normally distributed) and advanced linear
#' regression with Chi-Squared distribution (\code{distribution="dchisq"}, when the data is
#' distributed as Chi^2, for example squared normal standard errors).
#'
#' The advisable error measures to use in the test are RelMAE and RelMSE, which are
#' unbiased and whose logarithms are symmetrically distributed (Davydenko & Fildes,
#' 2013). In fact RelMSE should have F-distribution with h and h degrees of freedom
#' and its logarithm is a log F distribution, because each MSE * h has chi-square(h)
#' (assuming that the forecast error is normal).
#'
#' As for RelMAE, its distribution is trickier, because each MAE has folded normal
#' distribution (assuming that the original error is normal) and their ratio is
#' something complicated, but tractable (Kim, 2006).
#'
#' Still, given large samples, the parameters of the regression on logarithms of
#' the both RelMAE and RelMSE should have normal distribution. Thus
#' \code{distribution="dnorm"} can be used in this case (see examples).
#'
#' The test is equivalent to nemenyi test, when applied to the ranks of the error
#' measures on large samples.
#'
#' There is also a \code{plot()} method that allows producing either "mcb" or "lines"
#' style of plot. This can be regulated via \code{plot(x, style="lines")}.
#
#' @param data Matrix or data frame with observations in rows and variables in
#' columns.
#' @param distribution Type of the distribution to use. If this is a clear forecast error,
#' then \code{"dnorm"} is appropriate, leading to a simple Gausian linear
#' regression. \code{"dfnorm"} would lead to a alm model with folded normal
#' distribution. Finally, \code{"dchisq"} would lead to the alm with Chi
#' squared distribution. This value is passed to \code{alm()} function.
#' @param level The width of the confidence interval. Default is 0.95.
#' @param style What style of plot to use after the calculations. This can be
#' either "MCB" style or "Vertical lines" one.
#' @param select What column of data to highlight on the plot. If NULL, then
#' the method with the lowest value is selected.
#' @param plot If \code{TRUE} then the graph is produced after the calculations.
#' You can also use plot method on the produced object in order to get the same
#' effect.
#' @param ... Other parameters passed to plot function
#
#' @return If \code{plot=TRUE}, then the function plots the results after all
#' the calculations. In case of \code{distribution="dnorm"}, the closer to zero the
#' intervals are, the better model performs. When \code{distribution="dfnorm"} or
#' \code{distribution="dchisq"}, the smaller, the better.
#'
#' Function returns a list of a class "rmc", which contains the following
#' variables:
#' \itemize{
#' \item{mean}{Mean values for each method.}
#' \item{interval}{Confidence intervals for each method.}
#' \item{vlines}{Coordinates used for style="l", marking the groups of methods.}
#' \item{groups}{The table containing the groups. \code{TRUE} - methods are in the
#' same group, \code{FALSE} - they are not.}
#' \item{p.value}{p-value for the test of the significance of the model. This is a
#' log-likelihood ratios chi-squared test, comparing the model with the one with
#' intercept only.}
#' \item{importance}{The weights of the estimated model in comparison with the
#' model with the constant only. 0 means that the constant is better, 1 means that
#' the estimated model is the best.}
#' \item{level}{Significance level.}
#' \item{model}{lm model produced for the calculation of the intervals.}
#' \item{style}{Style of the plot to produce.}
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
#' ourData[,2] <- ourData[,2]+1
#' ourData[,3] <- ourData[,3]+0.7
#' ourData[,4] <- ourData[,4]+0.5
#' colnames(ourData) <- c("Method A","Method B","Method C - long name","Method D")
#' rmc(ourData, distribution="dnorm", level=0.95)
#
#' # In case of AE-based measures, distribution="dfnorm" should be selected
#' rmc(abs(ourData), distribution="dfnorm", level=0.95)
#'
#' # In case of SE-based measures, distribution="dchisq" should be selected
#' rmc(ourData^2, distribution="dchisq", level=0.95)
#'
#' # APE-based measures should not be used in general...
#'
#' # If RelMAE or RelMSE is used for measuring data, then it makes sense to use
#' # distribution="dnorm" and provide logarithms of the RelMAE, which can be approximated by
#' # normal distribution
#' ourData <- abs(ourData)
#' rmc(ourData / ourData[,1], distribution="dnorm", level=0.95)
#'
#' # The following example should give similar results to nemenyi test on
#' # large samples, which compares medians of the distributions:
#' rmc(t(apply(ourData,1,rank)), distribution="dnorm", level=0.95)
#'
#' @importFrom stats pchisq
#' @export rmc
rmc <- function(data, distribution=c("dnorm","dfnorm","dchisq"),
                level=0.95, style=c("mcb","lines"), select=NULL, plot=TRUE, ...){

    distribution <- distribution[1];
    style <- substr(style[1],1,1);

    if(is.data.frame(data)){
        data <- as.matrix(data);
    }
    dataNew <- as.vector(data);
    obs <- nrow(data);
    nMethods <- ncol(data);
    obsAll <- length(dataNew);
    namesMethods <- colnames(data);

    # Form the matrix of dummy variables
    xreg <- matrix(0,obsAll,nMethods,dimnames=list(NULL,namesMethods));
    for(i in 1:nMethods){
        xreg[obs*(i-1)+1:obs,i] <- 1;
    }
    # Collect stuff to form the data.frame
    dataNew <- as.data.frame(cbind(dataNew,xreg));
    colnames(dataNew)[1] <- "y";

    # This is the model used for the confidence intervals calculation
    lmModel <- alm(y~., data=dataNew[,-2], distribution=distribution);

    # Construct intervals
    lmCoefs <- coef(lmModel);
    # Force confint to be estimated inside the function
    lmIntervals <- confint(lmModel, level=level)[,-1];
    names(lmCoefs)[1] <- colnames(dataNew)[2];
    rownames(lmIntervals)[1] <- colnames(dataNew)[2];
    lmCoefs[-1] <- lmCoefs[1] + lmCoefs[-1];
    lmIntervals[-1,] <- lmCoefs[1] + lmIntervals[-1,];

    # Stuff needed for the importance of the model
    lmModel2 <- alm(y~1,data=dataNew, distribution=distribution);

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

    # Remove `` symbols in case of spaces in names
    names(lmCoefs) <- gsub("[[:punct:]]", "", names(lmCoefs));
    rownames(lmIntervals) <- names(lmCoefs);

    ### Prepare things for the groups for "lines" plot
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

    returnedClass <- structure(list(mean=lmCoefs, interval=lmIntervals, vlines=vlines, groups=groups,
                                    importance=importance, p.value=p.value, level=level, model=lmModel,
                                    style=style, select=select, distribution=distribution),
                               class="rmc");
    if(plot){
        plot(returnedClass, ...);
    }
    return(returnedClass);
}

#' @importFrom graphics axis box
#' @export
plot.rmc <- function(x, ...){
    nMethods <- length(x$mean);
    namesMethods <- names(x$mean);

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
        args$main <- paste0("Importance of the model is ", format(round(x$importance,3),nsmall=3),".\n",
                         x$level*100,"% confidence intervals constructed.");
    }

    # Save the current par() values
    parDefault <- par(no.readonly=TRUE);
    parMar <- parDefault$mar;

    if(ncol(x$groups)>1){
        pointCol <- "#0C6385";
        lineCol <- "#0DA0DC";
    }
    else{
        pointCol <- "darkgrey";
        lineCol <- "grey";
    }

    if(("style" %in% argsNames)){
        style <- substr(args$style,1,1);
        args$style <- NULL;
    }
    else{
        style <- x$style;
    }

    if(style=="m"){
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

        if(all(parMar==(c(5,4,4,2)+0.1))){
            parMar <- c(2, 2, 0, 0) + 0.1;
        }
        if(args$main!=""){
            parMar <- parMar + c(0,0,4,0);
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
        for(i in 1:nMethods){
            lines(rep(i,2),x$interval[i,],col=lineCol,lwd=2);
        }
        points(x$mean, pch=19, col=pointCol);
        axis(1,c(1:nMethods),namesMethods);
        axis(2);
        box(which="plot", col="black");

        abline(h=x$interval[x$select,], lwd=2, lty=2, col="grey");
    }
    else if(style=="l"){
        vlines <- x$vlines;

        k <- nrow(vlines);
        colours <- c("#0DA0DC","#17850C","#EA3921","#E1C513","#BB6ECE","#5DAD9D");
        colours <- rep(colours,ceiling(k/length(colours)))[1:k];

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
        par(mar=parMar)

        # Use do.call to use manipulated ellipsis (...)
        do.call(plot, args);

        if(k>1){
            for(i in 1:k){
                lines(c(i,i), vlines[i,], col=colours[i], lwd = 2);
                lines(c(0,i), rep(vlines[i,1],times=2), col="gray", lty=2);
                lines(c(0,i), rep(vlines[i,2],times=2), col="gray", lty=2);
            }
        }
        else{
            lines(c(1,1), c(1,nMethods), col=lineCol, lwd = 2);
            lines(c(0,1), rep(vlines[1,1],times=2), col=lineCol, lty=2);
            lines(c(0,1), rep(vlines[1,2],times=2), col=lineCol, lty=2);
        }
        axis(2,c(1:nMethods),namesMethods,las=2);
    }

    par(parDefault)
}

#' @export
print.rmc <- function(x, ...){
    cat(paste0("Regression for Multiple Comparison with ",switch(x$distribution,
                                                                 "dnorm"="normal",
                                                                 "dfnorm"="folded normal",
                                                                 "dchisq"="Chi-Squared"),
               " distribution.\n"));
    cat(paste0("The siginificance level is ",(1-x$level)*100,"%\n"));
    cat(paste0("Number of observations is ",nobs(x$model)," and number of dummies is ",length(x$mean),"\n"));
    cat(paste0("Significance test p-value: ",round(x$p.value,5),"\n"));
}
