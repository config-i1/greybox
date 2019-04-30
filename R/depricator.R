depricator <- function(newValue, ellipsis){
    if(!is.null(ellipsis$silent)){
        warning("You have provided 'silent' parameter. This is deprecated. Please, use 'quite' instead.");
        return(ellipsis$silent);
    }
    else if(!is.null(ellipsis$style)){
        warning("You have provided 'style' parameter. This is deprecated. Please, use 'outplot' instead.");
        return(ellipsis$style);
    }
    if(!is.null(ellipsis$B)){
        warning("You have provided 'B' parameter. This is deprecated. Please, use 'parameters' instead.");
        return(ellipsis$B);
    }
    if(!is.null(ellipsis$checks)){
        warning("You have provided 'checks' parameter. This is deprecated. Please, use 'fast' instead.");
        return(!ellipsis$checks);
    }
    else if(!is.null(ellipsis$style)){
        warning("You have provided 'style' parameter. This is deprecated. Please, use 'outplot' instead.");
        return(ellipsis$style);
    }
    else if(!is.null(ellipsis$bruteForce)){
        warning("You have provided 'bruteForce' parameter. This is deprecated. Please, use 'bruteforce' instead.");
        return(ellipsis$bruteForce);
    }
    else{
        return(newValue);
    }
}
