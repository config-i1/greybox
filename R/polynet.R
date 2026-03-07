polynet <- function(y, X, polyOrder=NULL, ...){

    varNorm <- polynetNorm(y, X);
    yNew <- varNorm$y;
    XNew <- varNorm$X;

    polynet1 <- polynetLayer1(yNew, XNew);
    xregData <- polynetLayer2(yNew, B=coef(polynet1), yFitted=fitted(polynet1), polyOrder);

    almReturned <- alm(y~., xregData, ...);

    almReturned$other$layer1 <- polynet1;
    almReturned$other$norm <- list(yMean=varNorm$yMean, yMAD=varNorm$yMAD,
                                   XMean=varNorm$XMean[1,,drop=FALSE], XMAD=varNorm$XMAD[1,,drop=FALSE]);
    almReturned$other$polyOrder <- polyOrder;
    # Substitute with the original units
    almReturned$data[,1] <- y;
    almReturned$fitted <- polynetDeNorm(almReturned$fitted, varNorm$yMAD, varNorm$yMean);
    # Reclass
    class(almReturned) <- c("polynet",class(almReturned));

    # Fit and return the model
    return(almReturned);
}

# This estimates the linear regression
polynetLayer1 <- function(y, X){
    # Add intercept
    X <- cbind(1, X);

    # Fit the model, extract fitted values
    lmFit <- .lm.fit(as.matrix(X), y);
    yFitted <- y - residuals(lmFit);

    return(list(coefficients=coef(lmFit), fitted=yFitted));
}

# This gets the fitted values from the initial regression
polynetLayer1Fit <- function(X, B){
    # Add intercept
    X <- cbind(1, X);

    return(X %*% B);
}

# This is the polynomial fitter
polynetLayer2 <- function(y, B, yFitted, polyOrder=NULL){
    nVariables <- length(B)-1;
    obsInSample <- length(yFitted);

    if(is.null(polyOrder)){
        polyOrder <- nVariables;
    }

    # The main polynomial
    xregData <- matrix(1, obsInSample, polyOrder+1,
                       dimnames=list(NULL, c("y",paste0("poly", 1:polyOrder))));
    xregData[,1] <- y

    for(i in 1:polyOrder){
        xregData[,i+1] <- yFitted^i;
    }

    return(xregData);
}

# This function normalises both X and y
polynetNorm <- function(y=NULL, X){
    X <- as.matrix(X);
    nVariables <- ncol(X);
    obsInSample <- nrow(X);

    XMAD <- matrix(apply(abs(diff(X)), 2, mean, na.rm=TRUE),
                   obsInSample, nVariables, byrow=TRUE);

    XMean <- matrix(apply(X, 2, mean, na.rm=TRUE),
                    obsInSample, nVariables, byrow=TRUE);

    XNew <- (X - XMean) / XMAD;

    yMAD <- yNew <- yMean <- NULL;
    if(!is.null(y)){
        yMAD <- mean(abs(diff(y)), na.rm=TRUE);
        yMean <- mean(y, na.rm=TRUE);
        yNew <- (y - yMean) / yMAD;
    }

    return(list(y=yNew, X=XNew, XMAD=XMAD, yMAD=yMAD, XMean=XMean, yMean=yMean));
}

# This function normalises both X and y
polynetReNorm <- function(X, XMAD, XMean){
    h <- nrow(X);
    nVariables <- ncol(X);

    XNew <- (X - matrix(XMean,h,nVariables,byrow=TRUE)) / matrix(XMAD,h,nVariables,byrow=TRUE);
    return(XNew);
}

# This function denormalises y to get to the origina units
polynetDeNorm <- function(y, yMAD, yMean){

    yNew <- yMean + y * yMAD

    return(yNew);
}

# Predict method for polynet
predict.polynet <- function(object, newdata, ...){
    XNew <- polynetReNorm(newdata, object$other$norm$XMAD, object$other$norm$XMean);

    B <- object$other$layer1$coefficients;
    yFitted <- polynetLayer1Fit(XNew, B);
    xregData <- polynetLayer2(NA, B=B, yFitted=yFitted, object$other$polyOrder);

    prediction <- predict.alm(object, xregData, ...);

    prediction$mean <- polynetDeNorm(prediction$mean, object$other$norm$yMAD, object$other$norm$yMean);
    if(!is.null(prediction$lower)){
        prediction$lower <- polynetDeNorm(prediction$lower, object$other$norm$yMAD, object$other$norm$yMean);
    }
    if(!is.null(prediction$upper)){
        prediction$upper <- polynetDeNorm(prediction$upper, object$other$norm$yMAD, object$other$norm$yMean);
    }

    return(prediction);
}


# Example of application:
#
# xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
# xreg <- cbind(100+0.5*xreg[,1]-1.5*xreg[,2]+0.75*xreg[,2]^2+rlaplace(100,0,100),xreg)
# colnames(xreg) <- c("y","x1","x2")
# inSample <- xreg[1:80,]
# outSample <- xreg[-c(1:80),]
#
# test <- polynet(inSample[,1], inSample[,-1,drop=FALSE], fast=T)
#
# test2 <- predict(test, outSample[,-1,drop=FALSE])
# plot(test2)
# lines(xreg[,1])
