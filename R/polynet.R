polynet <- function(y, X, polyOrder=NULL, ...){
    # Add intercept
    X <- cbind(1, X);

    nVariables <- ncol(X)-1;
    obsInSample <- nrow(X);

    # Fit the model, extract fitted values
    lmFit <- .lm.fit(as.matrix(X), y);
    yFitted <- y - resid(lmFit);

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

    # Fit and return the model
    return(alm(y~., xregData, ...));
}
