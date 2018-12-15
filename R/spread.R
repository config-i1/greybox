#' Construct scatterplot / boxplots for the data
#'
#' Function constructs the plots depending on the types of variables in the provided
#' matrix / data.frame.
#'
#' If the variables are in metric scale, then the classical scatterplot is constructed.
#' If one of them is either integer (up to 10 values) or categorical (aka 'factor'),
#' then boxplot (with grey dots corresponding to mean values) is constructed. Finally,
#' for the two categorical variables the table plot is returned. All of this is packed
#' in a matrix.
#'
#' @template AICRef
#' @template author
#' @keywords plots graph
#'
#' @param data Either matrix or data.frame with the data.
#' @param histograms If \code{TRUE}, then the histrograms and barplots are produced on
#' the diagonal of the matrix.
#' @param legend If \code{TRUE}, then the legend for the tableplot is drawn.
#' @param log If \code{TRUE}, then the logarithms of all numerical variables are taken.
#' @param ... Other parameters passed to the plot function.
#'
#' @return Function does not return anything. It just plots things.
#'
#' @seealso \code{\link[graphics]{plot}, \link[base]{table}, \link[greybox]{tableplot}}
#'
#' @examples
#'
#' ### Simple example
#' spread(mtcars)
#' spread(mtcars,log=TRUE)
#'
#' @importFrom graphics barplot boxplot hist mtext text title
#' @importFrom stats formula
#' @export spread
spread <- function(data, histograms=FALSE, legend=FALSE, log=FALSE, ...){
    ellipsis <- list(...);

    if(is.null(ellipsis$main)){
        mainTitle <- "";
        omaValues <- c(2,3,3,2);
    }
    else{
        mainTitle <- ellipsis$main;
        omaValues <- c(2,3,5,2);
    }

    if(!histograms){
        omaValues[c(2,3)] <- omaValues[c(2,3)]-1;
        omaValues[c(4)] <- omaValues[4]-1;
    }

    # if(legend){
    #     omaValues[c(4)] <- omaValues[c(4)] + 6;
    # }

    if(!is.data.frame(data)){
        data <- as.data.frame(data);
    }

    nVariables <- ncol(data);
    if(nVariables>20){
        stop(paste0("Too many variables! I can't work in such conditions! Too much pressure! ",
                    "Can you, please, reduce the number of variables at least to 20?"),
             call.=FALSE);
    }

    numericData <- vector(mode="logical", length=nVariables);
    for(i in 1:nVariables){
        numericData[i] <- is.numeric(data[[i]]);
        if(numericData[i]){
            if(length(unique(data[[i]]))<=10){
                numericData[i] <- FALSE;
            }
        }
    }

    if(log){
        if(any(data[numericData]<=0)){
            warning("Some variables have non-positive data, so logarithms cannot be produced for them.",call.=FALSE);
            nonZeroData <- numericData;
            nonZeroData[numericData] <- apply(data[numericData]>0,2,all);
        }
        else{
            nonZeroData <- numericData;
        }
        data[nonZeroData] <- log(data[nonZeroData]);
        colnames(data)[which(nonZeroData)] <- paste0("log_",colnames(data)[which(nonZeroData)]);
    }
    variablesNames <- colnames(data);

    parDefault <- par(no.readonly=TRUE);

    if(nVariables==1){
        if(numericData[1]){
            hist(data[[1]], main=mainTitle);
        }
        else{
            barplot(table(data[[1]]), col="white");
        }
    }
    else{
        par(mfcol=c(nVariables,nVariables), mar=rep(0,4), oma=omaValues, xaxt="s",yaxt="s",cex.main=1.5);
        for(i in 1:nVariables){
            for(j in 1:nVariables){
                if(i==j){
                    if(histograms){
                        if(numericData[i]){
                            hist(data[[i]], main="", axes=FALSE);
                        }
                        else{
                            barplot(table(data[[i]]), main="", axes=FALSE, axisnames=FALSE, col="white");
                        }
                    }
                    else{
                        if(numericData[i]){
                            midPoint <- (max(data[[i]])+min(data[[i]]))/2;
                            plot(data[[i]], data[[i]], col="white", axes=FALSE);
                            text(midPoint,midPoint,variablesNames[i],cex=1.5);
                        }
                        else{
                            uniqueValues <- unique(data[[i]]);
                            midPoint <- (1+length(uniqueValues))/2;
                            plot(0, 0, col="white", axes=FALSE,
                                 xlim=c(0.5,length(uniqueValues)+0.5), ylim=c(0.5,length(uniqueValues)+0.5));
                            text(midPoint,midPoint,variablesNames[i],cex=1.5);
                        }
                    }
                }
                else{
                    if(numericData[i] & numericData[j]){
                        plot(data[[i]],data[[j]], main="", axes=FALSE);
                    }
                    else if(numericData[i]){
                        boxplot(as.formula(paste0(variablesNames[i],"~",variablesNames[j])),data,horizontal=TRUE, main="", axes=FALSE);
                        points(tapply(data[[i]],data[[j]],mean), c(1:length(unique(data[[j]]))), pch=19, col="darkgrey")
                    }
                    else if(numericData[j]){
                        boxplot(as.formula(paste0(variablesNames[j],"~",variablesNames[i])),data, main="", axes=FALSE);
                        points(tapply(data[[j]],data[[i]],mean), pch=19, col="darkgrey")
                    }
                    else{
                        tableplot(data[[i]],data[[j]], labels=FALSE, main="", axes=FALSE);
                    }
                }

                # Add axis and labels if this is the first element
                if(i==1){
                    if(histograms){
                        mtext(variablesNames[nVariables-j+1], side=2, at=(j-0.35)/nVariables, line=1, adj=1, outer=TRUE);
                    }
                    else{
                        if(numericData[j]){
                            axis(2);
                        }
                        else{
                            uniqueValues <- unique(data[[j]]);
                            axis(2, at=seq(1,length(uniqueValues),length.out=length(uniqueValues)),
                                 labels=sort(uniqueValues));
                        }
                    }
                }
                #
                #                 if(j==1){
                #                     # Add axis at the top
                #                     if(!histograms){
                #                         if(numericData[i]){
                #                             axis(3);
                #                         }
                #                         else{
                #                             uniqueValues <- sort(unique(data[[i]]));
                #                             axis(3,at=seq(1,length(uniqueValues),length.out=length(uniqueValues)),
                #                                  labels=uniqueValues);
                #                         }
                #                     }
                #                 }

                # Add axis, if this is the last element in the matrix
                if(i==nVariables & histograms){
                    if(numericData[j]){
                        axis(4);
                    }
                    else{
                        uniqueValues <- unique(data[[j]]);
                        axis(4, at=seq(1,length(uniqueValues),length.out=length(uniqueValues)),
                             labels=sort(uniqueValues));
                    }
                }

                box();
            }
            # Add axis at the bottom
            if(numericData[i]){
                axis(1);
            }
            else{
                uniqueValues <- sort(unique(data[[i]]));
                axis(1,at=seq(1,length(uniqueValues),length.out=length(uniqueValues)),
                     labels=uniqueValues);
            }
            if(histograms){
                mtext(variablesNames[i], side=3, at=(i-0.5)/nVariables, line=1, outer=TRUE);
            }
        }

        if(mainTitle!=""){
            title(main=mainTitle, outer=TRUE, line=3, cex.main=2);
        }

        # if(legend){
        #     par(fig=c(0, 1, 0, 0.5), oma=c(0, 0, 0, 1), mar=c(0, 0, 0, 0), new=TRUE);
        #     plot(0, 0, type='n', bty='n', xaxt='n', yaxt='n');
        #     legend("topright", legend=c(1,0.5,0), lwd=0, fill=c(rgb(0,0,0,1),rgb(0.5,0.5,0.5,1),rgb(1,1,1,1)),
        #            bty="n");
        # }
    }

    par(parDefault);
}
