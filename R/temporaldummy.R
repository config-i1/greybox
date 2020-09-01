#' Dummy variables for provided seasonality type
#'
#' Function generates the matrix of dummy variables for the months / weeks / days /
#' hours / minutes / seconds of year / month / week / day / hour / minute.
#'
#' The function extracts dates from the provided object and returns a matrix with
#' dummy variables for the specified frequency type, with the number of rows equal
#' to the length of the object + the specified horizon. If a numeric vector is provided
#' then it will produce dummies based on typical values (e.g. 30 days in month). So it
#' is recommended to use proper classes with this method.
#'
#' Several notes on how the dummies are calculated in some special cases:
#' \itemize{
#' \item In case of weeks of years, the first week is defined according to ISO 8601.
#' }
#'
#' Note that not all the combinations of \code{type} and \code{of} are supported. For
#' example, there is no such thing as dummies for months of week. Also note that some
#' combinations are not very useful and would take a lot of memory (e.g. minutes of year).
#'
#' The function will return all the dummy variables. If you want to avoid the dummy
#' variables trap, you will need to exclude one of them manually.
#'
#' If you want to have a different type of dummy variables, let me know, I will
#' implement it.
#'
#' @param object Either a ts / msts / zoo / xts / tsibble object or a vector
#' of dates.
#' @param type Specifies what type of frequency to produce. For example, if
#' \code{type="month"}, then the matrix with dummies for months of the year will
#' be created.
#' @param of Specifies the frequency of what is needed. Used together with \code{type}
#' e.g. \code{type="day"} and \code{of="month"} will produce a matrix with dummies
#' for days of month (31 dummies).
#' @param h If not \code{NULL}, then the function will produce dummies for this
#' set of observations ahead as well, binding them to the original matrix.
#'
#' @return Class "dgCMatrix" with all the dummy variables is returned in case of numeric
#' variable. Feel free to drop one (making it a reference variable) or convert the object
#' into matrix (this will consume more memory than the returned class). In other cases the
#' object of the same class as the provided is returned.
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[greybox]{xregExpander}, \link[greybox]{xregMultiplier},
#' \link[greybox]{outlierdummy}}
#'
#' @examples
#' # Generate matrix with dummies for a ts object
#' x <- ts(rnorm(100,100,1),frequency=12)
#' temporaldummy(x)
#'
#' # Generate matrix with monthly dummies for a zoo object
#' x <- as.Date("2003-01-01")+0:99
#' temporaldummy(x, type="month", of="year", h=10)
#'
#' @rdname temporaldummy
#' @importFrom Matrix sparse.model.matrix
#' @export
temporaldummy <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                    of=c("year","quarter","month","week","day","hour","minute"), h=0) UseMethod("temporaldummy")

#' @rdname temporaldummy
#' @export
temporaldummy.default <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                            of=c("year","quarter","month","week","day","hour","minute"), h=0){
    type <- match.arg(type);
    of <- match.arg(of);

    # All the observations to produce
    obsAll <- length(object)+h;

    # Define frequency (we don't know it in case of default class)
    dataFrequency <- switch(type, "quarter"=4,
                            "month"=switch(of, "year"=12, 3),
                            "week"=switch(of, "year"=52, "quarter"=13, 4),
                            "day"=switch(of, "year"=365, "quarter"=120, "month"=30, 7),
                            "hour"=switch(of, "year"=8760, "quarter"=2880, "month"=720,
                                          "week"=168, 24),
                            "halfhour"=switch(of, "year"=17520, "quarter"=5760, "month"=1440,
                                              "week"=336, 48),
                            "minute"=switch(of, "year"=525600, "quarter"=172800, "month"=43200,
                                            "week"=10080, "day"=1440, 60),
                            "second"=switch(of, "year"=31536000, "quarter"=10368000, "month"=2592000,
                                            "week"=604800, "day"=86400, "hour"=3600, 60));

    # Create factors
    factorVariable <- factor(rep(c(1:dataFrequency),ceiling(obsAll/dataFrequency))[1:obsAll],
                             levels=c(1:dataFrequency));

    # Do model matrix for sparse factors
    temporaldummy <- sparse.model.matrix(~factorVariable-1);
    colnames(temporaldummy) <- paste0(type,c(1:dataFrequency),"of",of);

    return(temporaldummy);
}

#' @export
temporaldummy.ts <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                             of=c("year","quarter","month","week","day","hour","minute"), h=0){

    # Define frequency (we don't know it in case of default class)
    dataFrequency <- frequency(object);

    type <- switch(as.character(dataFrequency), "12"="month", "4"="quarter", "52"="week", "7"="day", "24"="hour",
                   "48"="halfhour", "60"="minute", "second");
    of <- switch(as.character(dataFrequency), "52"=, "12"=, "4"="year", "7"="week", "24"=,
                 "48"="day", "60"="hour", "minute");

    return(ts(as.matrix(temporaldummy.default(object, type=type, of=of, h=h)), start=start(object), frequency=dataFrequency));
}

#' @export
temporaldummy.Date <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                         of=c("year","quarter","month","week","day","hour","minute"), h=0){
    type <- match.arg(type);
    of <- match.arg(of);

    # All the observations to produce
    obsAll <- length(object)+h;

    # Change the length of object if h>1
    if(h>0){
        object <- c(object,tail(object,1)+c(1:h)*diff(tail(object,2)));
    }

    # Vectors with dates / times of year / month / week etc
    if(type=="quarter"){
        dateFinal <- quarters(object);
    }
    else if(type=="month"){
        dateFinal <- dateMonthsOfYear <- months(object, abbreviate=TRUE);
        if(of=="quarter"){
            # First month of each quarter
            dateFinal[dateMonthsOfYear %in% c("Jan","Apr","Jul","Oct")] <- 1;
            dateFinal[dateMonthsOfYear %in% c("Feb","May","Aug","Nov")] <- 2;
            dateFinal[dateMonthsOfYear %in% c("Mar","Jun","Sep","Dec")] <- 3;
        }
    }
    else if(type=="week"){
        # Weeks, starting from 1 according to ISO 8601
        dateFinal <- as.numeric(strftime(object, format="%V"));

        if(of!="year"){
            warning(paste0("The option of=",of,"is not yet implemented. Returning weeks of year instead."),
                    call.=FALSE);
        }
        # if(of=="quarter"){
        #     # 1 - 31 day of each Month
        #     # dateDayOfMonth <- as.numeric(strftime(object, format="%e"));
        #     for(i in 1:31){
        #         dateFinal[dateMonthsOfYear %in% c("Jan","Apr","Jul","Oct")] <- i;
        #     }
        # }
        # else if(of=="month"){
        #     dateFinal[]
        # }
        #
        # dateWeeks <- as.POSIXlt(object)$mday;
        # dateDayOfWeek <- weekdays(object, abbreviate=TRUE);
        # dateDayOfWeek <- as.numeric(strftime(object, format="%u"));
    }
    else if(type=="day"){
        if(of=="year"){
            dateFinal <- as.numeric(strftime(object, format="%j"));
        }
        else if(of=="quarter"){
            # Months of years
            dateMonthsOfYear <- months(object, abbreviate=TRUE);

            # Calculate number of days in each previous month
            dateYears <- as.numeric(strftime(object, format="%Y"));
            monthsOfYearNumber <- as.numeric(strftime(object, format="%m")) - 1;
            monthsOfYearNumber[monthsOfYearNumber==0] <- 12;
            ndaysInMonths <- as.numeric(strftime(as.Date(paste(dateYears + monthsOfYearNumber %/% 12,
                                                               monthsOfYearNumber %% 12 + 1, "01", sep="-"))-1,"%d"));

            # Extract days of months
            dateFinal <- as.numeric(strftime(object, format="%d"));
            # count number of days since the Jan / Apr / Jul / Oct
            dateFinal[dateMonthsOfYear %in% c("Feb","May","Aug","Nov")] <-
                ndaysInMonths[dateMonthsOfYear %in% c("Feb","May","Aug","Nov")] +
                dateFinal[dateMonthsOfYear %in% c("Feb","May","Aug","Nov")];

            # Amend the number of days, so that it counts both two months
            monthsOfYearNumber[monthsOfYearNumber==12] <- 0;
            monthsOfYearNumber[] <- monthsOfYearNumber + 1;
            ndaysInMonths[] <- ndaysInMonths +
                as.numeric(strftime(as.Date(paste(dateYears + monthsOfYearNumber %/% 12,
                                                  monthsOfYearNumber %% 12 + 1, "01", sep="-"))-1,"%d"));
            # count number of days since the Jan / Apr / Jul / Oct
            dateFinal[dateMonthsOfYear %in% c("Mar","Jun","Sep","Dec")] <-
                ndaysInMonths[dateMonthsOfYear %in% c("Mar","Jun","Sep","Dec")] +
                dateFinal[dateMonthsOfYear %in% c("Mar","Jun","Sep","Dec")];
        }
        else if(of=="month"){
            dateFinal <- as.numeric(strftime(object, format="%d"));
        }
        else if(of=="week"){
            dateFinal <- weekdays(object, abbreviate=TRUE);
        }
    }
    else if(any(type==c("hour","halfhour","minute","second"))){
        warning("The Date class does not support hours, minutes and seconds. Use POSIXt instead.", call.=FALSE);
        return(NULL);
    }

    # Create factors
    factorVariable <- factor(dateFinal)[1:obsAll];

    # Do model matrix for sparse factors
    temporaldummy <- sparse.model.matrix(~factorVariable-1);
    colnames(temporaldummy) <- paste0(type,sort(unique(dateFinal)),"of",of);

    return(temporaldummy);
}

#' @export
temporaldummy.POSIXt <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                         of=c("year","quarter","month","week","day","hour","minute"), h=0){
    type <- match.arg(type);
    of <- match.arg(of);

    # All the observations to produce
    obsAll <- length(object)+h;

    # Change the length of object if h>1
    if(h>0){
        object <- c(object,tail(object,1)+c(1:h)*diff(tail(object,2)));
    }

    # Invoke the respective class for the Date
    if(any(type==c("month","quarter","week","day"))){
        return(temporaldummy(as.Date(object), type=type, of=of, h=h));
    }
    else{
        if(type=="hour"){
            if(any(of==c("year","quarter","month"))){
                if(of=="year"){
                    dateOf <- as.numeric(strftime(object, format="%Y"));
                }
                else if(of=="quarter"){
                    dateOf <- as.numeric(substr(quarters(object),2,2));
                }
                else if(of=="month"){
                    dateOf <- as.numeric(strftime(object, format="%m"));
                }

                # Calculate the number of hours between the "ofs"
                dateOfDiffs <- c(1,diff(dateOf));
                dateOfDiffs[length(dateOf)] <- 1;
                hoursInOf <- difftime(object[which(dateOfDiffs!=0)[-1]],
                                      object[which(dateOfDiffs!=0)[-sum(dateOfDiffs!=0)]],
                                      units="hours");

                # Use those numbers in order to create the numeric vector
                dateFinal <- vector("numeric",obsAll);
                dateFinal[1:hoursInOf[1]] <- 1:hoursInOf[1];
                for(i in 2:length(hoursInOf)){
                    dateFinal[1:hoursInOf[i]+hoursInOf[i-1]] <- 1:hoursInOf[i];
                }
            }
            else{
                if(of=="week"){
                    #### This stuff assumes that the data is in hours!!! Fix this!!! ####
                    warning("Please, note that the function currently only handles hour of week if the data is in hours.",
                            call.=FALSE);
                    # Days of week in years
                    dateDaysOfWeekFirst <- as.numeric(strftime(object, format="%u"))[1];
                    # Hours in a day
                    dateHoursOfDayFirst <- as.numeric(strftime(object, format="%H"))[1]+1;
                    # The starting index of the day of week / hour
                    dateStart <- dateDaysOfWeekFirst*24 + dateHoursOfDayFirst;
                    # There's 168 hours in a week, so repeat that stuff
                    # +1 is needed just in case, not to create a smaller object than needed
                    dateFinal <- rep(c(1:(7*24)),ceiling(obsAll/(7*24))+1)[dateStart+1:obsAll];
                }
                else if(of=="day"){
                    dateFinal <- as.numeric(strftime(object, format="%H"));
                }
            }
        }
        else if(type=="halfhour"){
            stop("Sorry, but the halfhour option is not available yet", call.=FALSE);
        }
        else if(type=="minute"){
            if(any(of==c("year","quarter","month"))){
                if(of=="year"){
                    dateOf <- as.numeric(strftime(object, format="%Y"));
                }
                else if(of=="quarter"){
                    dateOf <- as.numeric(substr(quarters(object),2,2));
                }
                else if(of=="month"){
                    dateOf <- as.numeric(strftime(object, format="%m"));
                }

                # Calculate the number of minutes between the "ofs"
                dateOfDiffs <- c(1,diff(dateOf));
                dateOfDiffs[length(dateOf)] <- 1;
                hoursInOf <- difftime(object[which(dateOfDiffs!=0)[-1]],
                                      object[which(dateOfDiffs!=0)[-sum(dateOfDiffs!=0)]],
                                      units="mins");

                # Use those numbers in order to create the numeric vector
                dateFinal <- vector("numeric",obsAll);
                dateFinal[1:hoursInOf[1]] <- 1:hoursInOf[1];
                for(i in 2:length(hoursInOf)){
                    dateFinal[1:hoursInOf[i]+hoursInOf[i-1]] <- 1:hoursInOf[i];
                }
            }
            else{
                if(of=="week"){
                    #### This stuff assumes that the data is in hours!!! Fix this!!! ####
                    warning("Please, note that the function currently only handles minutes of week if the data is in minutes.",
                            call.=FALSE);
                    # Days of week in years
                    dateDaysOfWeekFirst <- as.numeric(strftime(object, format="%u"))[1];
                    # Hours in a day
                    dateHoursOfDayFirst <- as.numeric(strftime(object, format="%H"))[1];
                    # The starting index of the day of week / hour
                    dateStart <- dateDaysOfWeekFirst*24 + dateHoursOfDayFirst*60 + dateMinutesOfHourFirst-1;
                    # There's 168 hours in a week, so repeat that stuff
                    # +1 is needed just in case, not to create a smaller object than needed
                    dateFinal <- rep(c(1:(7*24*60)),ceiling(obsAll/(7*24*60))+1)[dateStart+1:obsAll];
                }
                else if(of=="day"){
                    #### This stuff assumes that the data is in hours!!! Fix this!!! ####
                    warning("Please, note that the function currently only handles minutes of day if the data is in minutes.",
                            call.=FALSE);
                    # Hours in a day
                    dateHoursOfDayFirst <- as.numeric(strftime(object, format="%H"))[1];
                    # The starting index of the day of week / hour
                    dateMinutesOfHourFirst <- as.numeric(strftime(object, format="%M"))[1];
                    # The starting minute of hour
                    dateStart <- dateHoursOfDayFirst*60+dateMinutesOfHourFirst-1;
                    # There's 168 hours in a week, so repeat that stuff
                    # +1 is needed just in case, not to create a smaller object than needed
                    dateFinal <- rep(c(1:(24*60)),ceiling(obsAll/(24*60))+1)[dateStart+1:obsAll];
                }
                else if(of=="hour"){
                    dateFinal <- as.numeric(strftime(object, format="%M"));
                }
            }
        }
        # Else... this is seconds
        else{
            if(of=="minute"){
                    dateFinal <- as.numeric(strftime(object, format="%S"));
            }
            else{
                stop(paste0("Sorry, but the seconds option is not available yet for the ",of),
                     call.=FALSE);
            }
        }
    }

    # Create factors
    factorVariable <- factor(dateFinal)[1:obsAll];

    # Do model matrix for sparse factors
    temporaldummy <- sparse.model.matrix(~factorVariable-1);
    colnames(temporaldummy) <- paste0(type,sort(unique(dateFinal)),"of",of);

    return(temporaldummy);
}

#' @export
temporaldummy.zoo <- function(object, type=c("month","quarter","week","day","hour","halfhour","minute","second"),
                         of=c("year","quarter","month","week","day","hour","minute"), h=0){
    type <- match.arg(type);
    of <- match.arg(of);

    dates <- time(object);

    # Change the length of object if h>1
    if(h>0){
        dates <- c(dates,tail(dates,1)+c(1:h)*diff(tail(dates,2)));
    }

    # Invoke the respective class for the date / time
    return(zoo(as.matrix(temporaldummy(dates, type=type, of=of, h=h)), order.by=dates));
}
