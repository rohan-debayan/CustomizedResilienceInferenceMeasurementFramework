import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from pgmpy.estimators import MMHC, BDeuScore, K2Score, BicScore, ConstraintBasedEstimator
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

def cd(score_var):
    # Load the data
    score = pd.read_csv("Input_CD.csv").dropna()

    # Convert all columns to numeric
    score = score.apply(pd.to_numeric, errors='coerce')

    # Normalize all columns (except score_var) to [0, 1]
    for col in score.columns:
        if col == score_var:
            continue
        scaler = MinMaxScaler()
        score[col] = scaler.fit_transform(score[col].values.reshape(-1, 1))
    
    # Split the dataset into training and test sets
    trainData, testData = train_test_split(score, test_size=0.2, random_state=42)
    
    # Create a blacklist to prevent any node from being a parent of score_var
    blacklist = [(node, score_var) for node in trainData.columns if node != score_var]
    
    # Hyperparameters to tune
    scoring_methods = [BDeuScore(trainData), K2Score(trainData), BicScore(trainData)]
    tabu_lengths = [5, 10, 15, 20]
    significance_levels = [0.01, 0.05, 0.1, 0.5]

    best_rmse = float('inf')
    best_hyperparameters = None

    # Hyperparameter tuning
    for scoring_method in scoring_methods:
        for tabu_length in tabu_lengths:
            for significance_level in significance_levels:
                
                # MMHC for structure learning with current hyperparameters
                est = MMHC(trainData, scoring_method=scoring_method)
                best_model = est.estimate(tabu_length=tabu_length, significance_level=significance_level)
                
                # Parameter learning using MLE for the Bayesian Network
    model = BayesianModel(best_model.edges())
    model.fit(trainData, estimator=MaximumLikelihoodEstimator)
    
    # Compute RMSE for each node using the learned Bayesian Network
    performance_df = pd.DataFrame(columns=['Node', 'RMSE'])
    for node in score.columns:
        if node == score_var:
            continue
        
        # Use the parents of the node in the Bayesian Network as features for linear regression
        parents = model.get_parents(node)
        
        # If a node has no parents, skip it
        if not parents:
            continue
        
        X_train = trainData[parents]
        y_train = trainData[node]
        X_test = testData[parents]
        y_test = testData[node]
        
        # Linear regression
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        predictions = lin_reg.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        performance_df = performance_df.append({'Node': node, 'RMSE': rmse}, ignore_index=True)

        # Check if current RMSE is the best
        current_rmse = performance_df['RMSE'].mean()
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_hyperparameters = (scoring_method, tabu_length, significance_level)

    print(f"Best RMSE: {best_rmse}")
    print(f"Best Hyperparameters: {best_hyperparameters}")
    return best_rmse,best_hyperparameters