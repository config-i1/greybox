association <- function(x, y=NULL, use=c("complete.obs","everything","all.obs","na.or.complete")){
    # Function returns the measures of association between the variables based on their type

    # If both variables are numeric, use cor() function
    # If both are nominal, then use Cramers V
    # If both are ordinal, use Kendall
    # If one is numeric, one is categorical, use expand and .lm.fit
    #
    #
    # if(is.factor(data)){
    #     if(any(sapply(data, is.ordered))){
    #         association <- cor(method="kendall")
    #     }
    #     else{
    #         association <- cramer()
    #     }
    # }
    # else{
    #     association <- cor;
    # }
    #
    # If one is numeric and the other one is categorical, us Intraclass correlation
    #
    # xregFactor <- model.matrix(~newFactor, data=x)
    #
    # test <- .lm.fit(as.matrix(xregFactor),y)
    # sqrt(1 - sum(residuals(test)^2) / sum((y-mean(y))^2))
}
