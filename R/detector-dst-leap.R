#' DST and Leap year detector functions
#'
#' Functions to detect, when Daylight Saving Time and leap year start and finish
#'
#' The \code{detectdst} function detects, when the change for the DST starts and ends. It
#' assumes that the frequency of data is not lower than hourly.
#' The \code{detectleap} function does similar for the leap year, but flagging the 29th
#' of February as a starting and to the 28th of February next year as the ending dates.
#'
#' In order for the methods to work, the object needs to be of either zoo / xts or POSIXt
#' class and should contain valid dates.
#'
#' @param object Either a zoo / xts object or a vector of dates / times in POSIXt / Date
#' class. Note that in order for \code{detectdst()} to work correctly, your data should
#' not have missing observations. Otherwise it might not be possible to locate, when DST
#' happens.
#'
#' @return List containing:
#' \itemize{
#' \item start - data frame with id (number of observation) and the respective dates,
#' when the DST / leap year start;
#' \item end - data frame with id and dates, when DST / leap year end.
#' }
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[greybox]{xregExpander}, \link[greybox]{temporaldummy},
#' \link[greybox]{outlierdummy}}
#'
#' @examples
#' # Generate matrix with monthly dummies for a zoo object
#' x <- as.POSIXct("2004-01-01")+0:(365*24*8)*60*60
#' detectdst(x)
#' detectleap(x)
#'
#' @rdname detectdst
#' @export
detectdst <- function(object) UseMethod("detectdst")

#' @export
detectdst.default <- function(object){
    warning("This function only works with classes zoo and POSIXt. Cannot extract DST observations.",call.=FALSE);
    return(NULL);
}

#' @export
detectdst.POSIXt <- function(object){
    # Function detects the dst days and flags them.
    # It returns a matrix with coordinates of the days that contain the change
    DSTIssues <- FALSE;

    # The Spring day
    DSTTime1 <- which(diff(as.numeric(strftime(object,"%H")))==2)+1;
    # If there are non-spring times, this might be an issue with the data
    if(any(!(strftime(object[DSTTime1],"%m") %in% c("03","04")))){
        DSTTime1 <- DSTTime1[strftime(object[DSTTime1],"%m") %in% c("03","04")];
        DSTIssues[] <- TRUE;
    }

    # The Autumn day
    DSTTime2 <- which(diff(as.numeric(strftime(object,"%H")))==0)+1;
    # If there are non-spring times, this might be an issue with the data
    if(any(!(strftime(object[DSTTime2],"%m") %in% c("10","11")))){
        DSTTime2 <- DSTTime2[strftime(object[DSTTime2],"%m") %in% c("10","11")];
        DSTIssues[] <- TRUE;
    }

    if(DSTIssues){
        warning("It seems that you have missing observations. The function might return wrong dates.",
                call.=FALSE);
    }

    return(list(start=data.frame(id=DSTTime1,date=object[DSTTime1]),
                end=data.frame(id=DSTTime2,date=object[DSTTime2])))
}

#' @export
detectdst.zoo <- function(object){
    return(detectdst.POSIXt(time(object)));
}


#' @rdname detectdst
#' @export
detectleap <- function(object) UseMethod("detectleap")

#' @export
detectleap.default <- function(object){
    warning("This function only works with classes zoo, Date and POSIXt. Cannot extract DST observations.",call.=FALSE);
    return(NULL);
}

#' @export
detectleap.Date <- function(object){
    uniqueLeapYears <- unique(strftime(object[which(strftime(object,"%m/%d")=="02/29")],"%Y"));
    LeapStart <- which(strftime(object,"%Y/%m/%d") %in% paste0(uniqueLeapYears,"/02/29"));
    LeapEnd <- LeapStart+366;

    return(list(start=data.frame(id=LeapStart,date=object[LeapStart]),
                end=data.frame(id=LeapEnd,date=object[LeapEnd])))
}

#' @export
detectleap.POSIXt <- function(object){
    # Get the years with 29th February
    uniqueLeapYears <- unique(strftime(object[which(strftime(object,"%m/%d")=="02/29")],"%Y"));
    LeapStartAll <- which(strftime(object,"%Y/%m/%d") %in% paste0(uniqueLeapYears,"/02/29"));
    uniqueYears <- unique(strftime(object[LeapStartAll],"%Y"));
    # Create the start for each years
    LeapStart <- vector("numeric",length(uniqueYears));
    for(i in 1:length(uniqueYears)){
        LeapStart[i] <- LeapStartAll[strftime(object[LeapStartAll],"%Y")==uniqueYears[i]][1];
    }
    # Create end dates, based on that
    LeapEndDate <- as.POSIXlt(object[LeapStart]);
    LeapEndDate$year <- LeapEndDate$year+1;
    LeapEnd <- which(strftime(object,"%Y/%m/%d:%H:%M:%S") %in% strftime(LeapEndDate,"%Y/%m/%d:%H:%M:%S"));


    return(list(start=data.frame(id=LeapStart,date=object[LeapStart]),
                end=data.frame(id=LeapEnd,date=object[LeapEnd])))
}

#' @export
detectleap.zoo <- function(object){
    return(detectleap.POSIXt(time(object)));
}
