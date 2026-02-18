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

# Function to calculate the dynamic multipliers.
# Do this recursively to avoid complications?
# dynmult <- function(object, parm){
#     ariPolynomial <- object$other$polynomial
#     coefDyn <- coef(object)[parm]
# }
