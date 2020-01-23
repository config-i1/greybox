identifyID <- function(y, ic="AIC"){

    if(all(y!=0) | all(y==0)){
        return(FALSE);
    }

    # First model with normal distribution
    model1 <- alm(y~1, data.frame(y=y), distribution="dnorm");
    # The second model with IG
    model2 <- alm(y~1, data.frame(y=y), distribution="dinvgauss", occurrence="plogis")
    # Fix the number of parameters
    model2$df <- 3;

    return(AIC(model2)<AIC(model1))
}
