identifyID <- function(y, ic=c("AIC","AICc","BIC","BICc")){

    # Select IC
    ic <- match.arg(ic);
    IC <- switch(ic,"AIC"=AIC,"BIC"=BIC,"BICc"=BICc,AICc);

    if(all(y!=0) | all(y==0)){
        return(FALSE);
    }

    # First model with normal distribution
    model1 <- alm(y~1, distribution="dnorm");
    # The second model with IG
    model2 <- alm(y~1, distribution="dinvgauss", occurrence="plogis");
    # Fix the number of parameters
    model2$df <- 3;

    return(IC(model2)<IC(model1));
}
