# library(greybox)
# library(smooth)
# library(tsutils)

# Created by: Kandrika Pritularga
# Date: 2nd March 2023
# Last edited: 2nd March 2023

# roLambda is a function to find the optimal hyperparameter
# for the implementation of ETS with smoothing parameter shrinkage.
# It implements a grid search for a sequence of numbers, between 0 and 1.
# There are two choices in finding the sequence: using logarithmic or arithmetic sequence.
# The former puts heavy weights on a small hyperparameter while the latter creates
# an equal sequence of numbers. The function offers two error measures: MSE and MAE.
#
# Parameters
# 1. model: a model from adam() function (what class is it?)
# 2. folds: how many origins to calculate the average of error measure across origins
# 3. nLambda: how many lambda we would like to have in the sequence
# 4. CVmeasure: error measure to be calculated to assess the best lambda
# 5. seqLambda: "log" is a logarithmic sequence and "arith" is an arithmetic sequence
# 6. outplot: whether to produce a plot, by default, FALSE


# Function
# roLambda <- function(model, folds = 5, nLambda = 100, loss = c("RIDGE", "LASSO"),
#                      CVmeasure = c("MSE", "MAE"),
#                      seqLambda = c("log", "arith"),
#                      outplot = FALSE) {
#
#   # create a sequence of lambda
#   if (seqLambda == "log") {
#     loghi <- log(0.99)
#     loglo <- log(0.01)
#     logrange <- loghi - loglo
#     interval <- -logrange/(nLambda - 1)
#     lambda <- exp(seq.int(from = loghi, to = loglo, by = interval))
#   } else {
#     lambda <- seq(0.01, 0.99, length.out = nLambda)
#   }
#
#   # getting details from the model
#   freq <- frequency(model$data)
#   modelStruc <- substr(model$model, 5, nchar(model$model)-1)
#   end.date <- time(model$data)[length(model$data)]
#   start.date <- end.date - folds/freq
#
#   # matrix to collect CVerror for each lambda
#   lambdaCVerror <- matrix(NA, ncol = nLambda, nrow = 2,
#                           dimnames = list(c("lambda", "CVmean"), 1:nLambda))
#
#   # looping for each lambda
#   for (l in 1:nLambda) {
#
#     # collect CVmean and CVse for each lambda
#     CVerror <- matrix(NA, ncol = folds, nrow = 1)
#     parameter <- matrix(NA, ncol = folds, nrow = length(model$B))
#     for (i in 1:folds) {
#       # create a time series for each origin
#       yOrigin <- window(model$data, end = start.date + (i-1)/freq)
#       # fit a model with shrinkage estimator using adam function
#       fitOrigin <- adam(yOrigin, model = modelStruc, loss = loss, lambda = lambda[l])
#
#       # collect y_{origin,t+1} and produce 1-step ahead forecast for each origin
#       yCV <- as.numeric(window(model$data, start = start.date + (i)/freq, end = start.date + (i)/freq))
#       fCV <- as.numeric(forecast(fitOrigin, h = 1)$mean)
#
#       # calculate the CV error measure
#       if (CVmeasure == "MSE") {
#         CVerror[,i] <- (yCV - fCV)^2
#       } else if (CVmeasure == "MAE") {
#         CVerror[,i] <- abs((yCV - fCV))
#       } else {
#         CVerror[,i] <- (yCV - fCV)^2
#       }
#
#       parameter[,i] <- fitOrigin$B
#
#     }
#
#     lambdaCVerror[1,l] <- lambda[l]
#     lambdaCVerror[2,l] <- mean(CVerror)
#     # lambdaCVerror[3,l] <- sd(CVerror)
#
#   }
#
#   # create a plot from the function
#   if (outplot) {
#
#     main.title <- paste(model$model, "-", loss)
#
#     if (seqLambda == "log") {
#
#       plot(log(lambdaCVerror[1,]), lambdaCVerror[2,], type = "o", pch = 20, col = "red",
#            ylab = "CV error", xlab = expression(log(lambda)), main = main.title)
#
#     } else {
#
#       plot(lambdaCVerror[1,], lambdaCVerror[2,], type = "o", pch = 20, col = "red",
#            ylab = "CV error", xlab = expression(lambda), main = main.title)
#
#     }
#   }
#
#   return(list(lambda = lambdaCVerror[1,],
#               CVmean = lambdaCVerror[2,],
#               lambda.min = lambdaCVerror[1, which.min(lambdaCVerror[2,])],
#               parameter = parameter))
#
# }
#
#
# ## Steps in analysis
# # 0. Preparation
# y <- sim.es(model = "ANN", obs = 48, frequency = 12, persistence = c(0.5), initial = c(100))
# plot(y)
#
# yTrain <- window(y$data, end = c(3,12))
# yTest <- window(y$data, start = c(4,1))
#
# ## Steps in analysis
# # 1. Fit a model without any shrinkage
# fit <- adam(yTrain, model = "ANN", loss = "MSE")
#
# # 2. Find the "optimal" lambda
# try.arith <- roLambda(fit, folds = 5, nLambda = 20,
#                       loss = "RIDGE", CVmeasure = "MSE",
#                       seqLambda = "arith", outplot = FALSE)
#
# # 3. Use the optimal lambda to fit a ETS with shrinkage
# modelStruc <- substr(fit$model, 5, nchar(fit$model)-1)
# fit.ridge <- adam(yTrain, model = modelStruc, loss = "RIDGE", lambda = try.arith$lambda.min)
#
# # 4. Produce forecasts
# fcst.ridge <- forecast(fit.ridge, h = length(yTest))$mean
# fcst.fit <- forecast(fit, h = length(yTest))$mean
#
# # 5. Compare the smoothing parameters
# c(fit.ridge$persistence, fit$persistence)

