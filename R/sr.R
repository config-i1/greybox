# Create a dataset from the provided data and models
#
# @export
folder <- function(...){
    # Function creates a dataset from the provided objects and models

    ellipsis <- list(...);
    listLength <- length(ellipsis);

    # Identify types of objects
    listDataFrame <- sapply(ellipsis, is.data.frame);
    listMatrix <- sapply(ellipsis, is.matrix);
    listVector <- sapply(ellipsis, is.vector);
    listModel <- !listDataFrame & !listMatrix & !listVector;

    # Create a vector of data types
    dataType <- rep("model",listLength);
    dataType[listDataFrame] <- "data.frame";
    dataType[listMatrix] <- "matrix";
    dataType[listVector] <- "vector";

    # Get the maximum number of observations
    obs <- max(unlist(c(sapply(ellipsis[listDataFrame], nrow),
                        sapply(ellipsis[listMatrix], nrow),
                        sapply(ellipsis[listVector], length),
                        sapply(ellipsis[listModel], nobs))))

    nVariables <- sum(unlist(c(sapply(ellipsis[listDataFrame], ncol),
                               sapply(ellipsis[listMatrix], ncol),
                               sum(listVector),
                               sapply(ellipsis[listModel], nvariate))));

    variablesNames <- vector("character",nVariables);
    variablesNumber <- vector("numeric",nVariables);

    # for(i in 1:nVariables){
    #     variablesNames[i]
    # }

    dataCreated <- as.data.frame(matrix(NA,obs,listLength));

    for(i in 1:listLength){
        dataCreated[,i]
    }
    ourReturn <- list(dataType);
    # Returned list:
    # 1. data (combined fitted and actuals)
    # 2. list of objects
    # ourReturn <- list(data=ourData, objects=ellipsis);
    return(structure(ourReturn,class="folder"));
}

# #' Stochastic Regressors Model
# #'
# #' @export
# srm <- function(formula, data, subset, na.action,
#                 # distribution=c("dnorm","dlaplace","ds","dgnorm","dlogis","dt","dalaplace",
#                 #                "dlnorm","dllaplace","dls","dlgnorm","dbcnorm","dfnorm",
#                 #                "dinvgauss","dgamma","dexp",
#                 #                "dpois","dnbinom",
#                 #                "dbeta","dlogitnorm",
#                 #                "plogis","pnorm"),
#                 # loss=c("likelihood","MSE","MAE","HAM","LASSO","RIDGE"),
#                 # occurrence=c("none","plogis","pnorm"),
#                 # scale=NULL,
#                 # orders=c(0,0,0),
#                 parameters=NULL, fast=FALSE, ...){
#     # Function estimates the regression model based on the provided data
#     # This one is analogous to alm(), but works with folder class
#
#     # Returned list:
#     # 1. data
#     # 2. formula
#     # 3. coef
#     # 4. folder - to generate point forecasts and variance
#     return(structure(ourReturn,class=c("srm","alm","greybox")));
# }
#
# #' @export
# forecast.srm <- forecast(object, newdata=NULL, h=NULL, ...){
#     # Function produces point forecasts and prediction intervals for SRM
#
# }
#
#
# #' @export
# print.folder <- function(x, ...){}
#
# #' @export
# print.srm <- function(x, ...){}
#

#### These will be moved to isFunction.R
# #' @rdname isFunctions
# #' @export
# is.folder <- function(x){
#     return(inherits(x,"folder"))
# }
#
# #' @rdname isFunctions
# #' @export
# is.srm <- function(x){
#     return(inherits(x,"srm"))
# }
