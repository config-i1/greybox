# ccd <- function(B, BUpper, BLower, CF, opts, ...){
#     # Function implements a simple Cyclical Coordinate Descent
#
#     CFLocal <- function(b){
#         B[j] <- b;
#         return(CF(B, ...));
#     }
#
#     niter <- 10
#     for(i in 1:niter){
#         for(j in 1:length(B)){
#             res <- optimise(CFLocal, lower=BLower, upper=BUpper, tol=1e-2)
#             B[j] <- res$minimum;
#             # res <- nloptr(B[j], CFLocal,
#             #               opts=opts,
#             #               lb=BLower[j], ub=BUpper[j]);
#             # B[j] <- res$solution;
#             CFValue <- res$objective;
#         }
#     }
#     return(list(solution=B,objective=CFValue));
# }
