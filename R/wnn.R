gower <- function(x, y, range=NULL){
    # The function compares two scalars, given the range and returns a value between 0 and 1
    if(is.factor(x) && is.factor(y)){
        return((x==y)*1);
    }
    else{
        return((x-y)/range);
    }
}

wnn <- function(formula, data){

    cl <- match.call();

    #### Form the necessary matrices ####
    # Call similar to lm in order to form appropriate data.frame
    mf <- match.call(expand.dots = FALSE);
    m <- match(c("formula", "data", "subset", "na.action"), names(mf), 0L);
    mf <- mf[c(1L, m)];
    mf$drop.unused.levels <- TRUE;
    mf[[1L]] <- quote(stats::model.frame);


    # If data is provided explicitly, check it
    if(exists("data",inherits=FALSE,mode="numeric") || exists("data",inherits=FALSE,mode="list")){
        if(!is.data.frame(data)){
            data <- as.data.frame(data);
        }
        else{
            dataOrders <- unlist(lapply(data,is.ordered));
            # If there is an ordered factor, remove the bloody ordering!
            if(any(dataOrders)){
                data[dataOrders] <- lapply(data[dataOrders],function(x) factor(x, levels=levels(x), ordered=FALSE));
            }
        }
        mf$data <- data;

        # If there are NaN values, remove the respective observations
        if(any(sapply(mf$data,is.nan))){
            warning("There are NaN values in the data. This might cause problems. Removing these observations.", call.=FALSE);
            NonNaNValues <- !apply(sapply(mf$data,is.nan),1,any);
            # If subset was not provided, change it
            if(is.null(mf$subset)){
                mf$subset <- NonNaNValues
            }
            else{
                mf$subset <- NonNaNValues & mf$subset;
            }
            dataContainsNaNs <- TRUE;
        }
        else{
            dataContainsNaNs <- FALSE;
        }
    }
    else{
        dataContainsNaNs <- FALSE;
    }

    responseName <- all.vars(formula)[1];

    dataWork <- eval(mf, parent.frame());
    y <- dataWork[,1];

    # Create a model from the provided stuff. This way we can work with factors
    # dataWork <- model.matrix(dataWork,data=dataWork);
    # dataWork[,1] <- y;
    # colnames(dataWork)[1] <- responseName;

    return(structure(list(data=dataWork),class="wnn"));
}

predict.wnn <- function(object, newdata=NULL, ...){
    if(is.null(newdata)){
        matrixOfxreg <- object$data;
        newdataProvided <- FALSE;
        # The first column is the response variable. Either substitute it by ones or remove it.
    }
    else{
        newdataProvided <- TRUE;

        if(!is.data.frame(newdata)){
            if(is.vector(newdata)){
                newdataNames <- names(newdata);
                newdata <- matrix(newdata, nrow=1, dimnames=list(NULL, newdataNames));
            }
            newdata <- as.data.frame(newdata);
        }
        else{
            dataOrders <- unlist(lapply(newdata,is.ordered));
            # If there is an ordered factor, remove the bloody ordering!
            if(any(dataOrders)){
                newdata[dataOrders] <- lapply(newdata[dataOrders],function(x) factor(x, levels=levels(x), ordered=FALSE));
            }
        }

        # Extract the formula and get rid of the response variable
        testFormula <- formula(object);
        testFormula[[2]] <- NULL;
        # Expand the data frame
        newdataExpanded <- model.frame(testFormula, newdata);
    }

    nRows <- nrow(matrixOfxreg);
    nCols <- ncol(matrixOfxreg);
    obsInSample <- nobs(object);

    # Prepare stuff for calculating weights
    ourWeights <- matrix(0,nRows,obsInSample);
    factors <- sapply(object$data, is.factor);
    ranges <- vector("numeric",nCols);
    ranges[!factors] <- apply(object$data[,-1,drop=FALSE][,!factors,drop=FALSE], 2, range);
    for(i in 1:nRows){
        for(j in 1:obsInSample){
            ourWeights[i,j] <- gower(object$data[,-1], matrixOfxreg[i,j]);
        }
    }
    ourForecast
}
