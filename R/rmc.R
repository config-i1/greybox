#' RMC test
#'
#' RMC stands for "Regression for Methods Comparison". This is a parametric
#' test for the comparison of medians of several distributions
#
#' This test is a parametric counterpart of nemenyi / MCB test and uses
#' asymptotic properties of regression models.
#'
#' The advisable error measures to use for data are RelMAE and RelMSE, which are
#' unbiased and have good properties. See examples for more details on how to
#' use them.
#
#' @param data Matrix or data frame with observations in rows and variables in
#' columns.
#' @param value Type of the provided value. If this is a clear forecast error,
#' then \code{"normal"} is appropriate, leading to a simple Gausian linear
#' regression. \code{"absolute"} would lead to truncated regression (if the
#' package "truncreg" is installed. Otherwise this will be a linear regression).
#' Finally, \code{"squared"} would lead to the glm with Gamma distribution.
#' @param level The width of the confidence interval. Default is 0.95.
#' @param sort If \code{TRUE} function sorts the final values of mean ranks.
#' If plots are requested via \code{type} parameter, then this is forced to
#' \code{TRUE}.
#' @param style What style of plot to use after the calculations. This can be
#' either "MCB" style or "Vertical lines" one.
#' @param select What column of data to highlight on the plot. If NULL, then
#' the method with the lowest value is selected.
#' @param plot If \code{TRUE} then the graph is produced after the calculations.
#' You can also use plot method on the produced object in order to get the same
#' effect.
#' @param ... Other parameters passed to plot function
#
#' @return Function returns the following variables:
#' \itemize{
#' \item{mean}{Mean rank of each method.}
#' \item{interval}{rmc intervals for each method.}
#' \item{p.value}{Friedman test p-value.}
#' \item{level}{Singificance level.}
#' \item{model}{lm model produced for the calculation of the intervals.}
#' \item{style}{Style of the plot to produce.}
#' \item{select}{The selected variable to highlight.}
#' }
#
#' @keywords htest
#' @template author
#
#' @references \itemize{
#' \item ???
#' }
#
#' @examples
#
#' N <- 50
#' M <- 4
#' ourData <- matrix(rnorm(N*M,mean=0,sd=1), N, M)
#' ourData[,2] <- ourData[,2]+1
#' ourData[,3] <- ourData[,3]+0.7
#' ourData[,4] <- ourData[,4]+0.5
#' colnames(ourData) <- c("Method A","Method B","Method C - long name","Method D")
#' rmc(ourData, value="n", level=0.95)
#
#' par(mar=c(2,0,2,0),cex=1.5)
#' rmc(ourData, level=0.95)
#'
#' # In case of AE-based measures, value="a" should be selected
#' rmc(abs(ourData), value="a", level=0.95)
#'
#' # In case of SE-based measures, value="s" should be selected
#' rmc(ourData^2, value="s", level=0.95)
#'
#' # APE-based measures should not be used in general...
#'
#' # If RelMAE or RelMSE is used for measuring data, then it makes sense to use
#' # value="n" and provide logarithms of the RelMAE, which will have asymptotic
#' # normal distribution
#' ourData <- abs(ourData)
#' ourData <- ourData / ourData[,1]
#' rmc(ourData, value="n", level=0.95)
#'
#' @importFrom stats pf glm Gamma
#' @export rmc
rmc <- function(data, value=c("normal","absolute","squared"),
                level=0.95, sort=TRUE, style=c("mcb","lines"),
                select=NULL, plot=TRUE, ...){

    value <- substr(value[1],1,1);
    if(!requireNamespace("truncreg", quietly = TRUE) & value=="a"){
        value <- "n";
        data <- log(data);
        warning(paste0("You use 'absolute' value, but don't have truncreg package installed. \n",
                       "The results of the test might be unreliable"), call.=FALSE);
    }
    style <- substr(style[1],1,1);

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
    if(value=="n"){
        lmModel <- lm(y~.-1,data=dataNew);
        lmCoefs <- coef(lmModel);
        lmIntervals <- confint(lmModel, level=level);
    }
    else if(value=="a"){
        lmModel <- truncreg::truncreg(y~.-1,data=dataNew);
        lmCoefs <- coef(lmModel);
        lmCoefs <- lmCoefs[-length(lmCoefs)]
        lmIntervals <- confint(lmModel, level=level)[names(lmCoefs),];
    }
    else if(value=="s"){
        lmModel <- glm(y~.-1,data=dataNew,family=Gamma);
        lmCoefs <- coef(lmModel);
        lmIntervals <- confint(lmModel, level=level);
    }

    # Model2 is needed for ANOVA results
    lmSummary <- summary(lm(y~.,data=dataNew));

    fStatistics <- pf(lmSummary$fstatistic[1],lmSummary$fstatistic[2],lmSummary$fstatistic[3],lower.tail=FALSE);

    if(is.null(select)){
        select <- which.min(lmCoefs);
    }

    if(sort){
        select <- which(namesMethods[order(lmCoefs)]==namesMethods[select]);
        lmIntervals <- lmIntervals[order(lmCoefs),];
        lmCoefs <- lmCoefs[order(lmCoefs)];
    }

    # Remove `` symbols in case of spaces in names
    names(lmCoefs) <- gsub("[[:punct:]]", "", names(lmCoefs));
    rownames(lmIntervals) <- names(lmCoefs);

    returnedClass <- structure(list(mean=lmCoefs, interval=lmIntervals, p.value=fStatistics, level=level,
                                    model=lmModel, style=style, select=select),
                               class="rmc");
    if(plot){
        plot(returnedClass);
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

    style <- x$style;

    if(x$p.value > 1-x$level){
        pointCol <- "darkgrey";
        lineCol <- "grey";
    }
    else{
        pointCol <- "#0C6385";
        lineCol <- "#0DA0DC";
    }

    if(x$style=="m"){
        plot(0,0, xlim=c(1,nMethods), ylim=range(x$interval), xaxt="n", xlab="", ylab="Values", pch=NA,
             main=paste0("P-value for ANOVA is ", format(round(x$p.value,3),nsmall=3),".\n",
                         x$level*100,"% confidence intervals constructed."));
        for(i in 1:nMethods){
            lines(rep(i,2),x$interval[i,],col=lineCol,lwd=2);
        }
        points(x$mean, pch=19, col=pointCol)
        axis(1,c(1:nMethods),namesMethods);

        abline(h=x$interval[x$select,], lwd=2, lty=2, col="grey");
    }
    else if(x$style=="l"){
        #Sort things just in case
        lmIntervals <- x$interval[order(x$mean),];
        lmCoefs <- x$mean[order(x$mean)];

        # Find groups
        vlines <- matrix(NA, nrow=nMethods, ncol=2);
        for(i in 1:nMethods){
            intersections <- which(!(lmIntervals[,2]<lmIntervals[i,1] | lmIntervals[,1]>lmIntervals[i,2]));
            vlines[i,] <- c(min(intersections),max(intersections));
        }

        # Get rid of duplicates and single member groups
        vlines <- unique(vlines);
        vlines <- vlines[apply(vlines,1,min) != apply(vlines,1,max),];
        # Re-convert to matrix if necessary and find number of remaining groups
        if(length(vlines)==2){
            vlines <- as.matrix(vlines);
            vlines <- t(vlines);
        }
        k <- nrow(vlines);
        colours <- c("#0DA0DC","#17850C","#EA3921","#E1C513","#BB6ECE","#5DAD9D");
        colours <- rep(colours,ceiling(k/length(colours)))[1:k];

        plot(0,0, ylim=c(1,nMethods), xlim=c(1,k), xaxt="n", yaxt="n", xlab="Groups", ylab="Methods", pch=NA,
             main=paste0("P-value for ANOVA is ", format(round(x$p.value,3),nsmall=3),".\n",
                         x$level*100,"% confidence intervals constructed."));
        if(k>1){
            for(i in 1:k){
                lines(c(i,i), vlines[i,], col=colours[i], lwd = 4);
                lines(c(0,i), rep(vlines[i,1],times=2), col="gray", lty=2);
                lines(c(0,i), rep(vlines[i,2],times=2), col="gray", lty=2);
            }
        }
        else{
            lines(c(1,1), c(1,nMethods), col="gray", lwd = 4);
            lines(c(0,1), rep(vlines[1,1],times=2), col="gray", lty=2);
            lines(c(0,1), rep(vlines[1,2],times=2), col="gray", lty=2);
        }
        axis(1,c(1:k),c(1:k));
        axis(2,c(1:nMethods),namesMethods);
    }
}
