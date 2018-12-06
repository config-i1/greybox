#' Construct a plot for categorical variable
#'
#' Function constructs a plot for two categorical variables based on table function
#'
#' The returned plot corresponds to the values of \code{table()} function with
#' density corresponding to the frequency of appearance. If the value appears more
#' often than the other (e.g. 0.5 vs 0.15), then it will be darker. The frequency of 0
#' corresponds to the white colour, the frequency of 1 corresponds to the black.
#'
#' @template author
#' @template keywords
#'
#' @param x First categorical variable.
#' @param y Second categorical variable.
#' @param labels Whether to print table labels inside the plot or not.
#' @param ... Other parameters passed to the plot function.
#'
#' @return Function does not return anything. It just plots things.
#'
#' @seealso \code{\link[graphics]{plot}, \link[base]{table}, \link[greybox]{spread}}
#'
#' @examples
#'
#' tableplot(mtcars$am, mtcars$gear)
#'
#' @export tableplot
tableplot <- function(x, y=NULL, labels=TRUE, ...){
    ellipsis <- list(...);

    if(is.null(y)){
        yIsProvided <- FALSE;
        y <- rep(0,length(x));
    }
    else{
        yIsProvided <- TRUE;
    }

    tableData <- table(x,y);
    tableData <- tableData / sum(tableData);

    # An option - make the same density or make it changable
    # tableDataColours <- 1 - 0.75 * tableData / max(tableData);
    tableDataColours <- 1 - tableData;

    xUnique <- sort(unique(x));
    yUnique <- sort(unique(y));
    xCoord <- seq(0.5,length(xUnique)+0.5,length.out=length(xUnique)+1);
    yCoord <- seq(0.5,length(yUnique)+0.5,length.out=length(yUnique)+1);
    xMid <- xCoord[-1]-0.5;
    yMid <- yCoord[-1]-0.5;

    if(is.null(ellipsis$main)){
        ellipsis$main <- "";
    }

    if(is.null(ellipsis$xlab)){
        xlab <- deparse(substitute(x));
        ellipsis$xlab <- "";
    }
    else{
        xlab <- ellipsis$xlab;
    }

    if(is.null(ellipsis$ylab)){
        ylab <- deparse(substitute(y));
        ellipsis$ylab <- "";
    }
    else{
        ylab <- ellipsis$ylab;
    }

    if(!is.null(ellipsis$axes)){
        axesValue <- ellipsis$axes;
    }
    else{
        axesValue <- TRUE;
    }
    ellipsis$axes <- FALSE;

    ellipsis$xlim <- range(xCoord);
    ellipsis$ylim <- range(yCoord);
    ellipsis$col <- "white";
    ellipsis$x <- 0;
    ellipsis$y <- 0;


    do.call(plot, ellipsis);
    for(i in 1:(length(xCoord)-1)){
        for(j in 1:(length(yCoord)-1)){
            polygon(c(c(xCoord[i],xCoord[i+1]),rev(c(xCoord[i],xCoord[i+1]))),
                    c(c(yCoord[j],yCoord[j]),c(yCoord[j+1],yCoord[j+1])),
                    col=rgb(tableDataColours[i,j],tableDataColours[i,j],tableDataColours[i,j]));
            if(labels){
                if(tableData[i,j]>0.5){
                    textCol <- "white";
                }
                else{
                    textCol <- "black";
                }
                text(xMid[i],yMid[j],labels=tableData[i,j],col=textCol);
            }
        }
    }
    if(axesValue){
        axis(1, at=xMid, labels=xUnique);
        if(yIsProvided){
            axis(2, at=yMid, labels=yUnique);
        }
    }
    box();
    title(xlab=xlab);
    if(yIsProvided){
        title(ylab=ylab);
    }
}
