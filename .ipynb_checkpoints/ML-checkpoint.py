from IPython.display import display
import ipywidgets as widgets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def split():
    test_split = widgets.FloatSlider(min=0.1, max=0.9, step=0.1, value=0.2, description='Test Split')
    random_state = widgets.IntText(value=42, description='Random State')

    def on_submit_button_clicked(b):
        print('Test Split:', test_split.value)
        print('Random State:', random_state.value)

    submit_button = widgets.Button(description='Submit')
    submit_button.on_click(on_submit_button_clicked)

    display(test_split)
    display(random_state)
    display(submit_button)
    
    return test_split.value, random_state.value

def split_data(X, y_class, yv_class, ya_class, yv_value, ya_value, yr_value, test_split, random_state):
    splits = {}

    # For Classifiers
    splits['X_train'], splits['X_test'], splits['y_train'], splits['y_test'] = train_test_split(X, y_class, test_size=test_split, random_state=random_state)
    splits['Xvc_train'], splits['Xvc_test'], splits['yvc_train'], splits['yvc_test'] = train_test_split(X, yv_class, test_size=test_split, random_state=random_state)
    splits['Xac_train'], splits['Xac_test'], splits['yac_train'], splits['yac_test'] = train_test_split(X, ya_class, test_size=test_split, random_state=random_state)

    # For Regressors
    splits['Xvr_train'], splits['Xvr_test'], splits['yvr_train'], splits['yvr_test'] = train_test_split(X, yv_value, test_size=test_split, random_state=random_state)
    splits['Xar_train'], splits['Xar_test'], splits['yar_train'], splits['yar_test'] = train_test_split(X, ya_value, test_size=test_split, random_state=random_state)
    splits['Xrr_train'], splits['Xrr_test'], splits['yrr_train'], splits['yrr_test'] = train_test_split(X, yr_value, test_size=test_split, random_state=random_state)
    
    return splits

def perform_linear_regression(splits, train_X_key, train_y_key, test_X_key, test_y_key, df):
    lin_reg = LinearRegression()
    params = {'normalize': [True, False]}
    grid_search = GridSearchCV(lin_reg, params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(splits[train_X_key], splits[train_y_key])

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    importances = best_model.coef_

    feature_importance_df = pd.DataFrame(importances, index=df.iloc[:,0:-8].columns, columns=['importance'])
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    ax = feature_importance_df.plot(kind='bar', legend=False, figsize=(10, 6), color='green', alpha=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Feature Importances')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.show()

    y_pred = best_model.predict(splits[test_X_key])
    mse = mean_squared_error(splits[test_y_key], y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(splits[test_y_key], y_pred)

    print(f'MSE: {mse:.5f}')
    print(f'RMSE: {rmse:.5f}')
    print(f'MAE: {mae:.5f}')

    return best_model, feature_importance_df

def perform_poly_regression(splits, train_X_key, train_y_key, test_X_key, test_y_key, df, degree_range):
    models = []
    mse_scores = []
    for degree in degree_range:
        model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression(normalize=True))
        model.fit(splits[train_X_key], splits[train_y_key])
        y_pred = model.predict(splits[test_X_key])
        mse = mean_squared_error(splits[test_y_key], y_pred)
        models.append(model)
        mse_scores.append(mse)

    best_index = np.argmin(mse_scores)
    best_model = models[best_index]
    best_degree = degree_range[best_index]

    print(f'Best Degree: {best_degree}')
    print(f'Minimum MSE: {mse_scores[best_index]:.5f}')

    # Original Feature Importance
    num_original_features = len(df.iloc[:,0:-8].columns)
    importances = best_model.named_steps['linearregression'].coef_
    poly_features = best_model.named_steps['polynomialfeatures']
    feature_names = poly_features.get_feature_names(input_features=df.iloc[:,0:-8].columns)
    importances = importances[:num_original_features]
    feature_names = feature_names[:num_original_features]
    feature_importance_df = pd.DataFrame(importances, index=feature_names, columns=['importance'])
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    ax = feature_importance_df.plot(kind='bar', legend=False, figsize=(10, 6), color='green', alpha=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Feature Importances')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.show()

    rmse = np.sqrt(mse_scores[best_index])
    mae = mean_absolute_error(splits[test_y_key], y_pred)
    print(f'MSE: {mse_scores[best_index]:.5f}')
    print(f'RMSE: {rmse:.5f}')
    print(f'MAE: {mae:.5f}')

    return best_model, feature_importance_df

def perform_ridge_regression(splits, train_X_key, train_y_key, test_X_key, test_y_key, df):
    param_grid = {'alpha': np.logspace(-3, 3, 7)}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')

    grid_search.fit(splits[train_X_key], splits[train_y_key])

    best_model = grid_search.best_estimator_

    importances = best_model.coef_

    feature_importance_df = pd.DataFrame(importances, index=df.iloc[:,0:-8].columns, columns=['importance'])
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    ax = feature_importance_df.plot(kind='bar', legend=False, figsize=(10, 6), color='green', alpha=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Feature Importances')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.show()
    
    y_pred = best_model.predict(splits[test_X_key])
    mse = mean_squared_error(splits[test_y_key], y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(splits[test_y_key], y_pred)

    print('MSE: {:.5f}'.format(mse))
    print('RMSE: {:.5f}'.format(rmse))
    print('MAE: {:.5f}'.format(mae))
    plt.show()

    return best_model, importances


def perform_svr(splits, train_X_key, train_y_key, test_X_key, test_y_key, df, param_grid, random_state):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    svr = SVR()

    grid_search = GridSearchCV(svr, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_search.fit(splits[train_X_key], splits[train_y_key].values.ravel())

    print("Best parameters:", grid_search.best_params_)
    print("Best score:", -grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(splits[test_X_key])

    mse = mean_squared_error(splits[test_y_key], y_pred)
    rmse = mean_squared_error(splits[test_y_key], y_pred, squared=False)
    mae = mean_absolute_error(splits[test_y_key], y_pred)
    r2 = r2_score(splits[test_y_key], y_pred)

    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R^2:", r2)

    results = permutation_importance(best_model, splits[test_X_key], splits[test_y_key], n_repeats=10, random_state=random_state)
    importances = results.importances_mean
    feature_importance_df = pd.DataFrame(importances, index=df.iloc[:,0:-8].columns, columns=['importance'])
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    ax = feature_importance_df.plot(kind='bar', legend=False, figsize=(10, 6), color='green', alpha=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title("Feature Importances")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.show()

    return best_model, importances

def perform_random_forest(splits, train_X_key, train_y_key, test_X_key, test_y_key, df, random_state):
    param_grid={'n_estimators': [50, 100, 200], 'max_depth': [10, 15, 20], 'min_samples_split': [15, 20, 25], 'min_samples_leaf': [4, 6, 8], 'max_features': ['auto', 'sqrt']}
    rf = RandomForestRegressor(random_state=random_state)

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                               n_iter=100, cv=5, verbose=0, random_state=random_state, n_jobs=-1)

    rf_random.fit(splits[train_X_key], splits[train_y_key].values.ravel())
    y_pred = rf_random.predict(splits[test_X_key])

    print('MSE:', mean_squared_error(splits[test_y_key], y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(splits[test_y_key], y_pred)))
    print('MAE:', mean_absolute_error(splits[test_y_key], y_pred))

    feature_importances = pd.DataFrame(rf_random.best_estimator_.feature_importances_,
                                       index = df.iloc[:,0:-8].columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances.index, feature_importances.importance, color='green')
    plt.gca().invert_yaxis()
    plt.title('Feature Importances', fontsize=16)
    plt.xlabel('Importance', ha='right')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    return rf_random.best_estimator_, feature_importances

def perform_xgb(splits, train_X_key, train_y_key, test_X_key, test_y_key, df):
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 7]
    }

    xgb = XGBRegressor(objective='reg:squarederror')

    grid_search = GridSearchCV(xgb, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(splits[train_X_key], splits[train_y_key])

    print('Best Hyperparameters: ', grid_search.best_params_)
    print('Best Score: ', grid_search.best_score_)

    xgb_best = XGBRegressor(objective='reg:squarederror', **grid_search.best_params_)

    xgb_best.fit(splits[train_X_key], splits[train_y_key], verbose=0)

    y_pred = xgb_best.predict(splits[test_X_key])

    # calculate performance metrics
    mse = mean_squared_error(splits[test_y_key], y_pred)
    rmse = mean_squared_error(splits[test_y_key], y_pred, squared=False)
    mae = mean_absolute_error(splits[test_y_key], y_pred)

    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('MAE: ', mae)

    # plot feature importance in a column chart
    feature_importances = pd.DataFrame(xgb_best.feature_importances_,
                                       index = df.iloc[:,0:-8].columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances.index, feature_importances.importance, color='green')
    plt.gca().invert_yaxis()
    plt.title('Feature Importances', fontsize=16)
    plt.xlabel('Importance', ha='right')
    plt.xticks(rotation=45, ha='right')
    plt.show()


    return xgb_best, feature_importances
