#' Create Lagged or Lead Variables for Time Series Regression
#'
#' `B()` acts as a backshift operator and creates lagged (past values) or
#' lead (future values) versions of a variable for use in regression formulas.
#' This function is designed to work within R formula syntax, similar
#' to how `I()` or `log()` work.
#'
#' The function calls for the `xregExpander()` to create lags/leads. So, you can pass
#' additional parameters to it via ellipsis.
#'
#' @param x A numeric vector or time series variable to be lagged or lead
#' @param k An integer specifying the lag order:
#'   \itemize{
#'     \item Positive values (e.g., `k = 1`) create lags (past values)
#'     \item Negative values (e.g., `k = -1`) create leads (future values)
#'     \item Zero (`k = 0`) returns the original variable unchanged
#'   }
#' @param ... Parameters passed to `xregExpander()`.
#'
#' @return A numeric vector of the same length as `x`. The missing values
#' are treated by the `xregExpander()`. By default they are extrapolated
#' (\code{gaps="auto"}).
#'
#' @details
#' When `k > 0` (lag), the function shifts values forward in time, so `B(x, 1)`
#' at time `t` contains the value of `x` at time `t-1`.
#'
#' When `k < 0` (lead), the function shifts values backward in time, so `B(x, -1)`
#' at time `t` contains the value of `x` at time `t+1`.
#'
#' @examples
#' # Create sample time series data
#' y = rnorm(10)
#'
#' # Create lags
#' B(y, 1)
#'
#' # Create leads
#' B(y, -1)
#'
#'
#' @seealso,
#' \code{\link[greybox]{xregExpander}} for data frame lag operations
#' \code{\link[stats]{lag}} for time series lag (different behavior)
#'
#' @export
B <- function(x, k, ...) {
    n <- length(x)
    if (k == 0) {
        return(x)
    }
    else{
        return(xregExpander(x, -k, ...)[,-1])
    }
}

# Dynamic multipliers for ARDL(p, k)
dynMultCalc <- function(phi, beta, h){
    # Function to calculate the dynamic multipliers.
    # Do this recursively to avoid complications?
    p <- length(phi)
    # treat beta_s = 0 for s > length(beta)-1
    max_beta_idx <- length(beta) - 1L

    cValues <- numeric(h)
    if (h < 1L) return(cValues)

    # s = 0
    cValues[1] <- beta[1]  # beta_0

    if (h == 1L) return(cValues)

    for (s in 1:(h - 1L)) {
        # beta_s (0 if s > max_beta_idx)
        beta_s <- if(s <= max_beta_idx) beta[s + 1L] else 0

        # sum_{i=1}^{min(p,s)} phi_i * m_{s-i}
        acc <- 0
        for (i in 1:min(p, s)) {
            acc <- acc + phi[i] * cValues[s - i + 1L]
        }
        cValues[s + 1L] <- beta_s + acc
    }
    cValues
}

# Extract beta parameter (lagged) for a specific variable
get_betas <- function(object, parm) {
    coefs <- coef(object)
    coef_names <- names(coefs)

    # Find the coefficient for the base variable (lag 0)
    # This matches "parm" exactly (not followed by comma)
    base_mask <- (coef_names == parm)

    # Find coefficients for lagged versions: B(parm, k)
    # Pattern: "B(varname, k)" where k is a number
    lag_pattern <- paste0("^B\\(", parm, ",\\s*([0-9]+)\\)$")
    lag_indices <- grep(lag_pattern, coef_names, perl = TRUE)

    # Extract lag orders from matched names
    lag_orders <- as.integer(gsub(lag_pattern, "\\1", coef_names[lag_indices]))

    # Create a data frame with lag orders and coefficients
    # Include base variable (lag 0)
    if (any(base_mask)) {
        betas_df <- data.frame(
            lag = 0,
            value = coefs[base_mask],
            stringsAsFactors = FALSE
        )
    } else {
        betas_df <- data.frame(
            lag = integer(0),
            value = numeric(0),
            stringsAsFactors = FALSE
        )
    }

    # Add lagged coefficients
    if (length(lag_indices) > 0) {
        lag_df <- data.frame(
            lag = lag_orders,
            value = coefs[lag_indices],
            stringsAsFactors = FALSE
        )
        betas_df <- rbind(betas_df, lag_df)
    }

    # Sort by lag order and return values as vector
    if (nrow(betas_df) == 0) {
        return(numeric(0))
    }

    betas_df <- betas_df[order(betas_df$lag), ]
    return(betas_df)
}


#' Dynamic multipliers from an ARDL model
#'
#' This function extracts the beta (distributed lag) coefficients for a specific
#' variable from an estimated ALM model and ARIMA(p,d,0) polynomials. It then uses
#' them to calculate dynamic multipliers for that variable for the horizon \code{h}.
#'
#' @param object An estimated ALM model object (with coefficients from \code{coef()}).
#' @param parm Character string of the variable name.
#' @param h Horizon for which to produce the dynamic multipliers.
#'
#' @return Numeric vector of dynamic multipliers over time
#'
#' @seealso,
#' \code{\link[greybox]{B}} for creating lagged variables
#'
#' @examples
#' \dontrun{
#'   # Fit a model with lagged variables
#'   test <- alm(drivers ~ kms + law + B(kms, 1) + B(kms, 2),
#'               Seatbelts, orders = c(1, 0, 0))
#'   multipliers(test, "kms", h=10)
#' }
#'
#' @export
multipliers <- function(object, parm, h=10){
    coefs <- coef(object);

    # Get positions of the parm in the vector of coefficients
    pos_kms <- which(grepl(paste0("\\b", parm, "\\b"), names(coefs)))

    if(length(pos_kms)>0){
        if(is.null(object$other$polynomial)){
            phi <- 0;
        }
        else{
            phi <- object$other$polynomial;
        }
        betas <- get_betas(object, parm);

        return(setNames(dynMultCalc(phi, betas[,2], h),
                        paste0("h",1:h)));
    }
    else{
        stop("The parameter \"",parm, "\" is not found in the model.");
    }
}
