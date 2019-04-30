utils::globalVariables(c("axis","box","colorRampPalette","brewer.pal",
                         "friedman.test","na.exclude","qtukey"));

# Nemenyi test
#
# Nemenyi is a non-parametric test for the comparison of medians of several
# distributions
#
# Hibon et al.(2012) demonstrated that Nemenyi test is equivalent to MCB
# (Multiple Comparison with the Best). The test ranks values of the variables
# and constructs confidence intervals. Based on these intervals the grouping of
# distributions is done, and a graph is produced so that it becomes apparent
# if there is a statistically significant difference between them.
#
# @param data Matrix or data frame with observations in rows and variables in
# columns.
# @param level The width of the confidence interval. Default is 0.95.
# @param sort If \code{TRUE} function sorts the final values of mean ranks.
# If plots are requested via \code{outplot} parameter, then this is forced to
# \code{TRUE}.
# @param outplot Type of the plot. This can be: 1. "none" - No plot;
# 2. "mcb" - MCB style plot; 3. "vmcb" - Vertical MCB style plot;
# 4. "line" - Line visualisation (ISF style), where numbers next to method
# names are the mean ranks; 5. "vline" - Vertical line visualisation.
# @param select Highlight the selected model (column). Number 1 to ncol. If
# \code{NULL}, then no highlighting is done.
# @param labels Optional labels for models. If \code{NULL} names of columns of
# \code{data} will be used. If number of labels != ncol, then it is assumed to
# be \code{NULL}.
# @param ... Other parameters passed to plot function.
#
# @return Function returns the following variables:
# \itemize{
# \item{mean}{Mean rank of each method.}
# \item{interval}{Nemenyi intervals for each method.}
# \item{p.value}{Friedman test p-value.}
# \item{hypothesis}{Friedman hypothesis result.}
# \item{CD}{Critical distance for nemenyi.}
# \item{level}{Significance level.}
# \item{nMethods}{Number of methods.}
# \item{obs}{Number of measurements.}
# }
#
# @keywords htest
# @template author
# @author Nikolaos Kourentzes
#
# @references \itemize{
# \item Koning, A. J., Franses, P. H., Hibon, M., & Stekler, H. O.
# (2005). The M3 competition: Statistical tests of the results.
# International Journal of Forecasting, 21(3), 397-409.
# \url{https://doi.org/10.1016/j.ijforecast.2004.10.003}
# \item Demsar, J. (2006). Statistical Comparisons of Classifiers over
# Multiple Data Sets. Journal of Machine Learning Research, 7, 1-30.
# \url{http://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf}
# \item Hibon M., Crone S. F., Kourentzes N., 2012. Statistical
# significance of forecasting methods: An empirical evaluation of the
# robustness and interpretability of the MCB, ANOM and Nemenyi tests,
# The 32nd Annual International Symposium on Forecasting, Boston.
# }
#
# @examples
#
# N <- 50
# M <- 4
# ourData <- matrix(rnorm(N*M,mean=0,sd=1), N, M)
# ourData[,2] <- ourData[,2]+1
# ourData[,3] <- ourData[,3]+0.7
# ourData[,4] <- ourData[,4]+0.5
# colnames(ourData) <- c("Method A","Method B","Method C - long name","Method D")
# nemenyi(ourData, level=0.95, outplot="vline")
#
# par(mar=c(2,0,2,0),cex=1.5)
# nemenyi(ourData,level=0.95,outplot="vline",main="",xlim=range(0,1.5))
#

# @importFrom grDevices colorRampPalette
# @importFrom graphics axis box
# @importFrom stats friedman.test na.exclude qtukey
# @importFrom RColorBrewer brewer.pal
#
# @export nemenyi
nemenyi <- function(data, level=0.95, sort=TRUE,
                    outplot=c("vline","none","mcb","vmcb","line"),
                    select=NULL, labels=NULL, ...)
{
    # Ivan Svetunkov, Nikolaos Kourentzes, 2014 - 2018.

    if(length(dim(data))!=2){
        stop(paste0("Data is of the wrong dimension! It should be a table with methods ",
                    "in columns and observations in rows."), call.=FALSE);
    }

    # In case data.frame was provided...
    data <- as.matrix(data);

    # Defaults
    outplot <- outplot[1]

    if(outplot!="none"){
        # Save the current par() values
        parDefault <- par(no.readonly=TRUE);
        parMar <- parDefault$mar;
        # If plot is asked, always sort the results
        sort <- TRUE
    }

    data <- na.exclude(data)
    obs <- nrow(data)
    nMethods <- ncol(data)

    # Check select < nMethods
    if(!is.null(select)){
        if (select > nMethods){
            select <- NULL
        }
    }

    # Checks for labels
    if(is.null(labels)){
        labels <- colnames(data)
        if (is.null(labels)){
            labels <- 1:nMethods
        }
    }
    else{
        labels <- labels[1:nMethods]
    }

    #### Test and statistic values ####
    # First run Friedman test. If insignificant then ignore Nemenyi (Weaker)
    friedmanPvalue <- friedman.test(data)$p.value
    if (friedmanPvalue <= 1-level){
        friedmanH <- "Ha: Different" # At least one method is different
    } else {
        friedmanH <- "H0: Identical" # No evidence of differences between methods
    }

    # Nemenyi critical distance and bounds of intervals
    tukeyQuant <- qtukey(level,nMethods,Inf) * sqrt((nMethods*(nMethods+1))/(12*obs))

    #### Ranks ####
    # Rank methods for each time series
    ranksMatrix <- matrix(NA, nrow=obs, ncol=nMethods)
    colnames(ranksMatrix) <- labels
    for (i in 1:obs){
        ranksMatrix[i, ] <- rank(data[i,],na.last="keep",ties.method="average")
    }

    # Calculate mean rank values
    ranksMeans <- colMeans(ranksMatrix)

    # Calculate intervals for each of the methods
    # 0.5 is needed in order to get to the half of the critical distance
    ranksIntervals <- rbind(ranksMeans - 0.5*tukeyQuant,ranksMeans + 0.5*tukeyQuant)
    colnames(ranksIntervals) <- labels

    # Sort interval matrix and means
    if(sort){
        orderIdx <- order(ranksMeans)
        ranksMeans <- ranksMeans[orderIdx]
        ranksIntervals <- ranksIntervals[,order(ranksIntervals[2,])]
        if (!is.null(labels)){
            labels <- labels[orderIdx]
        }
        if (!is.null(select)){
            select <- which(orderIdx == select)
        }
    }

    # Labels for plots
    if(is.null(labels)){
        labels <- colnames(ranksIntervals)
    }

    # Produce plots

    # # For all plots
    args <- list(...)
    argsNames <- names(args)

    if(!("xaxs" %in% argsNames)){
        args$xaxs <- "i"
    }
    if(!("yaxs" %in% argsNames)){
        args$yaxs <- "i"
    }

    args$x <- args$y <- NA
    args$axes <- FALSE

    #### MCB plot ####
    # MCB style plot
    if(outplot=="mcb"){
        cmp <- brewer.pal(3,"Set1")[1:2]
        # Choose colour depending on Friedman test result
        if (friedmanPvalue > 1-level){
            pcol <- "gray"
        } else {
            pcol <- cmp[2]
        }
        # Find min max
        yMax <- max(ranksIntervals)
        yMin <- min(ranksIntervals)
        yMax <- yMax + 0.1*(yMax-yMin)
        yMin <- yMin - 0.1*(yMax-yMin)

        if(!("main" %in% argsNames)){
            args$main <- paste0("Friedman: ", format(round(friedmanPvalue,3),nsmall=3),
                                " (", friedmanH, ") \n MCB interval: ",
                                format(round(tukeyQuant,3),nsmall=3))
        }
        if(!("xlab" %in% argsNames)){
            args$xlab <- ""
        }
        if(!("ylab" %in% argsNames)){
            args$ylab <- "Mean ranks"
        }
        # Remaining defaults
        if(is.null(args$xlim)){
            args$xlim <- c(0,nMethods+1)
        }
        if(is.null(args$ylim)){
            args$ylim <- c(yMin,yMax)
        }

        if(all(parMar==(c(5,4,4,2)+0.1))){
            parMar <- c(2, 2, 0, 0) + 0.1
        }
        if(!is.null(args$main)){
            if(args$main!=""){
                parMar <- parMar + c(0,0,4,0)
            }
        }
        par(mar=parMar)

        # Use do.call to use manipulated ellipsis (...)
        do.call(plot, args)
        # Plot rest
        points(1:nMethods, ranksMeans, pch=20, lwd=4)
        axis(1, at=c(1:nMethods), labels=labels)
        axis(2)
        box(which="plot", col="black")
        # Intervals for best method
        if(is.null(select)){
            select <- 1
        }
        lines(c(0,nMethods+1), rep(ranksIntervals[1,select],times=2), col="gray", lty=2)
        lines(c(0,nMethods+1), rep(ranksIntervals[2,select],times=2), col="gray", lty=2)
        # Intervals for all methods
        for(i in 1:nMethods){
            lines(rep(i,times=2), ranksIntervals[,i], type="b", lwd=2, col=pcol)
        }
        # Highlight identical
        idx <- !((ranksIntervals[2,select] < ranksIntervals[1,]) |
                     (ranksIntervals[1,select] > ranksIntervals[2,]))
        points((1:nMethods)[idx], ranksMeans[idx], pch=20, lwd=4, col=cmp[1])
    }
    #### Vertical MCB plot ####
    # MCB style plot - vertical
    else if(outplot=="vmcb"){
        cmp <- brewer.pal(3,"Set1")[1:2]
        # Find max label size
        labelSize <- max(nchar(labels))
        # Choose colour depending on Friedman test result
        if (friedmanPvalue > 1-level){
            pcol <- "gray"
        }
        else{
            pcol <- cmp[2]
        }
        # Find min max
        xMax <- max(ranksIntervals)
        xMin <- min(ranksIntervals)
        xMax <- xMax + 0.1*(xMax-xMin)
        xMin <- xMin - 0.1*(xMax-xMin)

        if(!("main" %in% argsNames)){
            args$main <- paste0("Friedman: ", format(round(friedmanPvalue,3),nsmall=3),
                                " (", friedmanH, ") \n MCB interval: ",
                                format(round(tukeyQuant,3),nsmall=3))
        }
        if(!("ylab" %in% argsNames)){
            args$ylab <- ""
        }
        if(!("xlab" %in% argsNames)){
            args$xlab <- "Mean ranks"
        }
        # Remaining defaults
        if(is.null(args$xlim)){
            args$xlim <- c(xMin,xMax)
        }
        if(is.null(args$ylim)){
            args$ylim <- c(0,nMethods+1)
        }
        # If the default mar is used, do things.
        if(all(parMar==(c(5,4,4,2)+0.1))){
            parMar <- c(2, labelSize/2, 0, 0) + 0.1
        }
        else{
            parMar <- parMar + c(2, labelSize/2, 0, 0)
        }
        if(!is.null(args$main)){
            if(args$main!=""){
                parMar <- parMar + c(0,0,4,0)
            }
        }
        par(mar=parMar)

        do.call(plot,args)
        # Plot rest
        points(ranksMeans,1:nMethods,pch=20,lwd=4)
        axis(2,at=c(1:nMethods),labels=labels,las=2)
        axis(1)
        box(which="plot", col="black")
        # Intervals for best method
        if (is.null(select)){
            select <- 1
        }
        lines(rep(ranksIntervals[1,select],times=2), c(0,nMethods+1), col="gray", lty=2)
        lines(rep(ranksIntervals[2,select],times=2), c(0,nMethods+1), col="gray", lty=2)
        # Intervals for all methods
        for (i in 1:nMethods){
            lines(ranksIntervals[,i], rep(i,times=2), type="b", lwd=2, col=pcol);
        }
        # Highlight identical
        idx <- !((ranksIntervals[2,select] < ranksIntervals[1,]) |
                     (ranksIntervals[1,select] > ranksIntervals[2,]))
        points(ranksMeans[idx], (1:nMethods)[idx], pch=20, lwd=4, col=cmp[1])
    }
    #### Line plot ####
    # Line style plot (as in ISF)
    else if(outplot == "line"){
        # Find groups
        rline <- matrix(NA, nrow=nMethods, ncol=2)
        for(i in 1:nMethods){
            tloc <- which((abs(ranksMeans-ranksMeans[i])<tukeyQuant) == TRUE)
            rline[i,] <- c(min(tloc),max(tloc))
        }
        # Get rid of duplicates and single member groups
        rline <- unique(rline)
        rline <- rline[apply(rline,1,min)!=apply(rline,1,max),]
        # Re-convert to matrix if necessary and find number of remaining groups
        if(length(rline)==2){
            rline <- as.matrix(rline)
            rline <- t(rline)
        }
        k <- nrow(rline)
        # Choose colour depending on Friedman test result
        cmp <- colorRampPalette(brewer.pal(12,"Paired"))(k)
        if (friedmanPvalue > 1-level){
            pcol <- rep("gray",times=k)
        }
        else{
            pcol <- cmp
        }
        # Prepare method labels and add mean rank to them
        lbl <- labels
        lblm <- matrix(NA,nrow=1,ncol=nMethods)
        labelSize <- matrix(NA,nrow=1,ncol=nMethods)
        for (i in 1:nMethods){
            if (is.null(lbl)){
                lblm[i] <- i
            } else {
                lblm[i] <- lbl[i]
            }
            lblm[i] <- paste(lblm[i]," - ",format(round(ranksMeans[i],2),nsmall=2),sep="")
            labelSize[i] <- nchar(lblm[i])
        }
        labelSize <- max(labelSize)
        # Produce plot
        if(!("main" %in% argsNames)){
            args$main <- paste0("Friedman: ", format(round(friedmanPvalue,3),nsmall=3),
                                " (", friedmanH, ") \n CD: ",
                                format(round(tukeyQuant,3),nsmall=3))
        }
        if(!("ylab" %in% argsNames)){
            args$ylab <- ""
        }
        if(!("xlab" %in% argsNames)){
            args$xlab <- ""
        }
        # Remaining defaults
        if(is.null(args$xlim)){
            args$xlim <- c(1,nMethods)
        }
        if(is.null(args$ylim)){
            args$ylim <- c(0,k+1)
        }

        # If the default mar is used, do things.
        if(all(parMar==(c(5,4,4,2)+0.1))){
            parMar <- c(labelSize/2, 2, 0, 2) + 0.1
        }
        else{
            parMar <- parMar + c(labelSize/2, 0, 0, 0)
        }
        if(!is.null(args$main)){
            if(args$main!=""){
                parMar <- parMar + c(0,0,4,0)
            }
        }
        par(mar=parMar)

        do.call(plot,args)
        points(1:nMethods,rep(0,nMethods),pch=20,lwd=4)
        if(k>0){
            for(i in 1:k){
                lines(rline[i,], c(i,i), col=pcol[i], lwd = 4)
                lines(rep(rline[i,1],times=2), c(0,i), col="gray", lty = 2)
                lines(rep(rline[i,2],times=2), c(0,i), col="gray", lty = 2)
            }
        }
        axis(1,at=c(1:nMethods),labels=lblm,las=2)
        if(!is.null(select)){
            points(select,0,pch=20,col=brewer.pal(3,"Set1")[1],cex=2)
        }
    }
    #### Vertical line plot ####
    # Line style plot (as in ISF) - vertical
    else if(outplot=="vline"){
        # Find groups
        rline <- matrix(NA, nrow=nMethods, ncol=2)
        for(i in 1:nMethods){
            tloc <- which((abs(ranksMeans-ranksMeans[i])<tukeyQuant) == TRUE)
            rline[i,] <- c(min(tloc),max(tloc))
        }
        # Get rid of duplicates and single member groups
        rline <- unique(rline)
        rline <- rline[apply(rline,1,min) != apply(rline,1,max),]
        # Re-convert to matrix if necessary and find number of remaining groups
        if(length(rline)==2){
            rline <- as.matrix(rline)
            rline <- t(rline)
        }
        k <- nrow(rline)
        # Choose colour depending on Friedman test result
        cmp <- colorRampPalette(brewer.pal(12,"Paired"))(k)
        if(friedmanPvalue > 1-level){
            pcol <- rep("gray",times=k)
        }
        else{
            pcol <- cmp
        }
        # Prepare method labels and add mean rank to them
        lbl <- labels
        lblm <- matrix(NA,nrow=1,ncol=nMethods)
        labelSize <- matrix(NA,nrow=1,ncol=nMethods)
        for(i in 1:nMethods){
            if(is.null(lbl)){
                lblm[i] <- i
            }
            else{
                lblm[i] <- lbl[i]
            }
            lblm[i] <- paste(lblm[i]," - ",format(round(ranksMeans[i],2),nsmall=2),sep="")
            labelSize[i] <- nchar(lblm[i])
        }
        labelSize <- max(labelSize)
        # Produce plot
        if(!("main" %in% argsNames)){
            args$main <- paste0("Friedman: ", format(round(friedmanPvalue,3),nsmall=3),
                                " (", friedmanH, ") \n CD: ",
                                format(round(tukeyQuant,3),nsmall=3))
        }
        if(!("ylab" %in% argsNames)){
            args$ylab <- ""
        }
        if(!("xlab" %in% argsNames)){
            args$xlab <- ""
        }
        # Remaining defaults
        if(is.null(args$xlim)){
            args$xlim <- c(0,k+1)
        }
        if(is.null(args$ylim)){
            args$ylim <- c(1,nMethods)
        }
        # If the default mar is used, do things.
        if(all(parMar==(c(5,4,4,2)+0.1))){
            parMar <- c(2, labelSize/2, 2, 0) + 0.1
        }
        else{
            parMar <- parMar + c(0, labelSize/2, 0, 0)
        }
        if(!is.null(args$main)){
            if(args$main!=""){
                parMar <- parMar + c(0,0,4,0)
            }
        }
        par(mar=parMar)

        do.call(plot,args)
        points(rep(0,nMethods),1:nMethods,pch=20,lwd=4)
        if(k>0){
            for(i in 1:k){
                lines(c(i,i), rline[i,], col=pcol[i], lwd = 4)
                lines(c(0,i), rep(rline[i,1],times=2), col="gray", lty = 2)
                lines(c(0,i), rep(rline[i,2],times=2), col="gray", lty = 2)
            }
        }
        axis(2,at=c(1:nMethods),labels=lblm,las=2)
        if(!is.null(select)){
            points(0,select,pch=20,col=brewer.pal(3,"Set1")[1],cex=2)
        }
    }

    # Revert to the original par() values
    if(outplot!="none"){
        par(parDefault)
    }

    return(structure(list("mean"=ranksMeans,"interval"=ranksIntervals,
                          "p.value"=friedmanPvalue,"hypothesis"=friedmanH,
                          "CD"=tukeyQuant,"level"=level,"nMethods"=nMethods,
                          "obs"=obs),
                     class="nemenyi"))
}

# @export
summary.nemenyi <- function(object,...){
    print(object)
}

# @export
print.nemenyi <- function(x,...){
    writeLines("Friedman and Nemenyi Tests")
    writeLines(paste0("The significance level is ", (1-x$level)*100, "%"))
    writeLines(paste0("Number of observations is ", x$obs, " and number of methods is ", x$nMethods))
    writeLines(paste0("Friedman test p-value: ", format(round(x$p.value,4),nsmall=4) , " - ", x$hypothesis))
    writeLines(paste0("Nemenyi critical distance: ", format(round(x$CD,4),nsmall=4)))
}
