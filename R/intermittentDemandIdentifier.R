idi <- function(y, ic=c("AIC","AICc","BIC","BICc")){

    # Select IC
    ic <- match.arg(ic);
    IC <- switch(ic,"AIC"=AIC,"BIC"=BIC,"BICc"=BICc,AICc);

    # Data for the two models model
    xreg1 <- data.frame(y=y, x=y)
    xreg1$x <- lowess(y)$y;

    # Datat for demand sizes
    xreg2Sizes <- data.frame(y=y, x=y)
    xreg2Sizes$x[] <- 0;
    xreg2Sizes$x[y!=0] <- lowess(y[y!=0])$y;

    # Data for demand occurrence
    xreg2Occurrence <- data.frame(y=y, x=y)
    xreg2Occurrence$y[] <- (xreg2Occurrence$y!=0)*1;
    xreg2Occurrence$x[] <- lowess(xreg2Occurrence$y)$y;

    # First model with rectified normal distribution
    model1 <- suppressWarnings(alm(y~., xreg1, distribution="drectnorm"));

    # Model for demand occurrence
    model2Occurrence <- suppressWarnings(alm(y~., xreg2Occurrence, distribution="plogis"));
    # The second model: mixture of Box-Cox Normal and Logistic
    model2 <- suppressWarnings(alm(y~., xreg2Sizes, distribution="dbcnorm", occurrence=model2Occurrence));

    # Fix the number of estimated parameters to include occurrence part
    # model2$df <- model2$df + nparam(model2$occurrence)
    # model2$df.residual <- model2$df.residual - nparam(model2$occurrence)

    return(structure(list(intermittent=IC(model2)<IC(model1), model1=model1, model2=model2),
                     class="idi"));
}

print.idi <- function(x, ...){
    if(x$intermittent){
        cat("Data is intermittent\n");
    }
    else{
        cat("Data is not intermittent\n");
    }
}
