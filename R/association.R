association <- function(x, y=NULL, use=c("complete.obs","everything","all.obs","na.or.complete")){
    # Function returns the measures of association between the variables based on their type

    use <- use[1];

    if(!is.matrix(x) & !is.data.frame(x)){
        nVariablesX <- 1;
        namesX <- substitute(x);
    }
    else{
        nVariablesX <- ncol(x);
        namesX <- colnames(x);
    }

    if(!is.matrix(y) & !is.data.frame(y)){
        nVariablesY <- 1;
        namesY <- substitute(x);
    }
    else{
        nVariablesY <- ncol(y);
        namesY <- colnames(y);
    }

    matrixAssociation <- matrix(0,nVariablesX,nVariablesY,dimnames=list(namesX,namesY));
    matrixPValues <- matrix(0,nVariablesX,nVariablesY,dimnames=list(namesX,namesY));

    numericDataX <- vector(mode="logical", length=nVariablesX);
    for(i in 1:nVariablesX){
        numericDataX[i] <- is.numeric(x[,i]);
        if(numericDataX[i]){
            if(length(unique(x[,i]))<=10){
                numericDataX[i] <- FALSE;
            }
        }
    }

    numericDataY <- vector(mode="logical", length=nVariablesY);
    for(i in 1:nVariablesY){
        numericDataY[i] <- is.numeric(y[,i]);
        if(numericDataY[i]){
            if(length(unique(y[,i]))<=10){
                numericDataY[i] <- FALSE;
            }
        }
    }

    for(i in 1:nVariablesX){
        for(j in 1:nVariablesY){
            if(i==j){
                matrixAssociation[i,j] <- 1;
                next;
            }

            if(numericDataX[i] & numericDataY[j]){
                matrixAssociation[i,j] <- cor(x[,i],y[,j],use=use);
                matrixPValues[i,j] <- cor.test(x[,i],y[,j])$p.value;
            }
            else if(!numericDataX[i] & !numericDataY[j]){
                cramerOutput <- cramer(x[,i],y[,j]);
                matrixAssociation[i,j] <- cramerOutput$value;
                matrixPValues[i,j] <- cramerOutput$p.value;
            }
            else if(!numericDataX[i] & numericDataY[j]){
                iccOutput <- icc(x[,i],y[,j]);
                matrixAssociation[i,j] <- iccOutput$value;
                matrixPValues[i,j] <- iccOutput$p.value;
            }
            else{
                iccOutput <- icc(y[,i],x[,j]);
                matrixAssociation[i,j] <- iccOutput$value;
                matrixPValues[i,j] <- iccOutput$p.value;
            }
        }
    }

    return(list(value=matrixAssociation,p.value=matrixPValues));
}
