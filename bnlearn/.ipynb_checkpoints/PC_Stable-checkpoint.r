pcstable <- function(p, test, alpha, score_var) {
  
  library("readxl")
  library(nonlinearICP)
  library(CondIndTests)
  library(bnlearn)
  library(ISLR)
  library(tidyr)
  library(dplyr)
  library(pracma)
  library(caret)

  score <- as.data.frame(read.csv("Input_CD.csv"))
  score <- na.omit(score)

  score <- lapply(score, as.numeric)

  min_max_df <- data.frame(matrix(ncol = 3, nrow = 0))
  colnames(min_max_df) <- c('factor', 'max', 'min')

  max_x <- c()
  min_x <- c()

  for (col in colnames(score)){
    if (col == score_var)
      next
    max_x <- append(max_x, max(score[, col]))
    min_x <- append(min_x, min(score[, col]))
  }

  min_max_norm <- function(x, min_x, max_x) {
       (x - min_x) / (max_x - min_x)
  }

  factors <- colnames(score)
  factors <- factors[factors != score_var]

  for (i in 1:length(factors)){
    min_max_df[i, 'factor'] = factors[i]
    min_max_df[i, 'max'] = max_x[i]
    min_max_df[i, 'min'] = min_x[i]
    
    score[, factors[i]] = min_max_norm(score[, factors[i]], min_max_df$min[i], min_max_df$max[i])
  }

  # Defining Blacklist Arcs
  black.list = matrix(c(rep(score_var, length(factors)), factors), ncol = 2, byrow = TRUE)
  colnames(black.list) <- c("from", "to")

  trainIndex <- createDataPartition(score[[score_var]], p, list = FALSE)
    
  trainData <- score[trainIndex, ]
  testData <- score[-trainIndex, ]

  suppressWarnings({bn <- boot.strength(trainData, R = 100, m = nrow(trainData), algorithm = "pc.stable", algorithm.args = list(blacklist = black.list, test='smc-zf', alpha = 0.1), debug = FALSE)})

  avg.diff = averaged.network(bn)

  fitted_bn <- bn.fit(avg.diff, data = score, method = "mle")

  coefficients_df <- do.call(rbind, lapply(fitted_bn, coef))
  
  parents_node = parents(avg.diff, score_var)
  spouses_node = spouses(avg.diff, score_var)
  causal_features <- union(parents_node, spouses_node)

  y <- score[, score_var]
  X <- score[, causal_features]
  LinRegData <- cbind(y, X)
  
  trainLinRegData <- LinRegData[trainIndex,]
  testLinRegData <- LinRegData[-trainIndex,]
  
  model <- lm(y~., data=as.data.frame(trainLinRegData))
  pred <- predict(model, newdata = as.data.frame(testLinRegData))

  diff <- testLinRegData[, 1] - pred
  rmse <- sqrt(mean(diff^2))
  mse <- mean(diff^2)
  mae <- mean(abs(diff))

  saveRDS(avg.diff, "model.rds")
  #avg.diff <- readRDS("model.rds")
  
  return(list(arcs = arcs(avg.diff), coefficients = coefficients_df, parents_node = parents_node, mae = mae, mse = mse, rmse = rmse))
}

  return(list(arcs = arcs(avg.diff), parents_node = parents_node, mae = mae, mse = mse, rmse = rmse))
}