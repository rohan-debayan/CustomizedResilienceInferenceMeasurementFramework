cd <- function(score_var) {
# Clearing the workspace
rm(list = ls())

# Load the necessary libraries
library("readxl")
library(nonlinearICP)
library(CondIndTests)
library(bnlearn)
library(ISLR)
library(tidyr)
library(dplyr)
library(pracma)
library(caret)

# Read the data
score <- as.data.frame(read.csv("Input_CD.csv"))
score <- na.omit(score)

# Convert all columns to numeric
score <- lapply(score, as.numeric)

# For score
# Normalizing Factors' Value
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

select.columns <- colnames(score)

tests <- c('cor','mc-cor','smc-cor','zf','mc-zf','smc-zf','mi-g','mc-mi-g','smc-mi-g','mi-g-sh')
alphas <- c(0.001,0.005,0.01,0.05,0.1)

performance_df <- data.frame(matrix(ncol = 3, nrow = 0)) 
colnames(performance_df) <- c('Test', 'Alpha', 'RMSE')

count <- 0
suppressWarnings({
  count <- count +124
  set.seed(count)
  
  trainIndex <- createDataPartition(score$YA_Value, p = 0.8, list = FALSE)
  
  trainData <- score[trainIndex, ]
  testData <- score[-trainIndex, ]
  
  ## List for storing the Mean RMSEs for all configurations
  mean_rmses <- c()
  
  for (test in tests){ # for all test types
    for (alpha in alphas){ # for all alpha types
      start_time <- Sys.time() # for computation time
      
      # Developing model
      bn <- pc.stable(trainData, blacklist = black.list, test = test, alpha = alpha, debug = FALSE, undirected = FALSE)
      
      # List for storing RMSEs for all nodes in the current iteration
      rmses <- c()
      
      # Iterating through every node to calculate the RMSE
      for (node in select.columns){
        
        ## Determining the Parents and Spouses Boundary of the node
        parents_node = parents(bn, node)
        spouses_node = spouses(bn, node)
        causal_features <- union(parents_node,spouses_node)
        if(node == 'YA_Value'){
          print(causal_features)
        }
        if(length(causal_features) == 0) # If no Markov Boundary
          next
        
        ## Creating temporary dataset for Linear Regression
        y <- score[, node]
        X <- score[, causal_features]
        LinRegData <- cbind(y, X)
       
        trainLinRegData <- LinRegData[trainIndex,]
        testLinRegData <- LinRegData[-trainIndex,]
       
        model <- lm(y~., data=as.data.frame(trainLinRegData)) # fit linear regression model based on markov boundary of the node
        pred <- predict(model, newdata = as.data.frame(testLinRegData))
         
        rmse <- sqrt(mean(((testLinRegData[, 1] - pred)^2)))
        if(is.na(rmse))
          next
        
        rmses <- append(rmses, rmse)
      }
      time_taken <- Sys.time() - start_time # for computation time
      performance_df[nrow(performance_df) + 1,] = c(test, alpha, mean(rmses), time_taken)
    }
  }
})

write.csv(performance_df, "PC_Stable_Score.csv", row.names=FALSE)
}