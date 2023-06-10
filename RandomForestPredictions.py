# -*- coding: utf-8 -*-
"""
RandomForestPredictions.py

Purpose:


Version:
    
    
Date:
    

Author:
    Eveline Wilgenkamp
"""

###########################################################
### IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import time
import os
from datetime import date, timedelta, datetime
from dateutil.parser import parse
from timer import Timer

from sklearn.ensemble import RandomForestRegressor

###########################################################
### WORKING DIRECTORY
os.chdir('C:\\Users\\eveli\\Documents\\ThesisData')

###########################################################
### DATA IMPORT (AND PREP)
dfData = pd.read_csv('jensen2022_data.csv', 
                     sep=',', 
                     index_col=False)

# Drop rows with missing excess returns
dfData.dropna(subset=['ret_exc'], inplace=True)

# Create year variable
dfData['eom'] = pd.to_datetime(dfData['eom'], format='%Y-%m-%d')
dfData['year'] = dfData['eom'].dt.year


###########################################################
### FUNCTIONS
def tuneRandomForest(X_train, y_train, X_valid, y_valid):
    # Initialize parameters to tune (similar to Gu et al)
    params_grid = {'max_depth':[1, 2, 3, 4, 5],
                   'max_features':[5, 10, 25, 50, 75, 100]}
    
    # Initializations the random forest regressor
    rfg = RandomForestRegressor(n_estimators=300,
                                random_state=69,
                                n_jobs=5)
    
    best_score = np.NINF
    best_params = None
    
    # Loop through the grid and find best combinations
    for max_depth in params_grid['max_depth']:
        for max_features in params_grid['max_features']:
            # Set new parameters
            rfg.set_params(max_depth=max_depth,
                           max_features=max_features)
            
            # Fit on the training data
            rfg.fit(X=X_train, y=y_train)
            
            # Evaluate performance on validation set
            score = rfg.score(X=X_valid, y=y_valid)
            
            # Store params if best model so far
            if score > best_score:
                best_score = score
                best_params = {'max_depth':max_depth, 'max_features':max_features}
                
    return best_params

def predictRandomForest(X_train, y_train, X_test, y_test, best_params):
    # Initializations the random forest regressor
    best_rfg = RandomForestRegressor(n_estimators=300,
                                     random_state=69,
                                     n_jobs=5,
                                     max_depth=best_params['max_depth'],
                                     max_features=best_params['max_features'])
    
    # Fit on the training set
    best_rfg.fit(X=X_train, y=y_train)
    
    # Predict on the test set (one year of observations)
    y_pred = best_rfg.predict(X_test)
    
    return y_pred

    
###########################################################
### PREDICTIONS
# Characteristics
lChars = ['sic', 'cowc_gr1a', 'oaccruals_at', 'oaccruals_ni', 
         'taccruals_at', 'taccruals_ni', 'fnl_gr1a', 'ncol_gr1a', 'nfna_gr1a', 
         'noa_at', 'aliq_at', 'at_gr1', 'be_gr1a', 'capx_gr1', 'coa_gr1a', 
         'col_gr1a', 'emp_gr1', 'inv_gr1', 'inv_gr1a', 'lnoa_gr1a', 
         'mispricing_mgmt', 'ncoa_gr1a', 'nncoa_gr1a', 'noa_gr1a',
         'ppeinv_gr1a', 'ret_60_12', 'sale_gr1', 'seas_2_5na', 'age', 
         'aliq_mat', 'at_be', 'bidaskhl_21d', 'cash_at', 'netdebt_me', 
         'tangibility', 'beta_60m', 'beta_dimson_21d', 'betabab_1260d',
         'betadown_252d', 'ivol_capm_21d', 'ivol_capm_252d', 'ivol_ff3_21d', 
         'rmax1_21d', 'rmax5_21d', 'rvol_21d', 'rvol_252d', 'turnover_126d', 
         'zero_trades_126d', 'zero_trades_21d', 'zero_trades_252d', 
         'prc_highprc_252d', 'ret_12_1', 'ret_3_1', 'ret_6_1', 'ret_9_1', 
         'seas_1_1na', 'ocf_at_chg1', 'ret_12_7',  'sale_emp_gr1', 
         'seas_1_1an', 'tax_gr1a', 'dolvol_var_126d', 'ebit_bev', 'ebit_sale', 
         'intrinsic_value', 'ni_be', 'o_score', 'ocf_at', 'ope_be', 'ope_bel1', 
         'turnover_var_126d', 'at_turnover', 'cop_at', 'cop_atl1', 'gp_at', 
         'gp_atl1', 'mispricing_perf', 'op_at', 'op_atl1', 'opex_at', 
         'qmj_prof', 'qmj_safety', 'sale_bev', 'corr_1260d', 'coskew_21d',
         'dbnetis_at', 'kz_index', 'lti_gr1a', 'pi_nix', 'seas_11_15an',
         'seas_11_15na', 'seas_2_5an', 'seas_6_10an', 'ami_126d', 
         'dolvol_126d', 'market_equity', 'prc', 'iskew_capm_21d',
         'iskew_ff3_21d', 'ret_1_0', 'rmax5_rvol_21d', 'rskew_21d', 'at_me', 
         'be_me', 'bev_mev', 'chcsho_12m', 'debt_me', 'div12m_me', 
         'ebitda_mev', 'eq_dur', 'eqnpo_12m', 'fcf_me', 'ni_me', 'ocf_me',
         'sale_me', 'seas_6_10na']

# Set initial train, valid, test split
train_start = 1960
train_end = 1978
valid_start = 1979
valid_end = 1990
test_start = 1991
test_end = 2020

# List to store the predictions in
lPred = []

# Expanding window loop
for year in range(test_start, test_end+1):
    print(f'Current prediction iteration: {year}')
    
    # Prepare train, validation, and test tests
    dfTrain = dfData[(dfData['year'] >= train_start) & (dfData['year'] <= train_end)]
    dfValid = dfData[(dfData['year'] >= valid_start) & (dfData['year'] <= valid_end)]
    dfTest = dfData[dfData['year'] == test_start]
    
    X_train = dfTrain[lChars]
    y_train = dfTrain['ret_exc']
    X_valid = dfValid[lChars]
    y_valid = dfValid['ret_exc']
    X_test = dfTest[lChars]
    y_test = dfTest['ret_exc']
    
    print(f'Tuning started for iteration: {year}')
    
    # Find the best parameters for the model refit
    best_params = tuneRandomForest(X_train, y_train, X_valid, y_valid)
    
    print(f'Tuning finished for iteration: {year}')
    
    print(f'Predictions started for iteration: {year}')
    
    # Fit the model for the prediction year at hand
    y_pred = predictRandomForest(X_train, y_train, X_test, y_test, best_params)
    
    print(f'Predictions finished for iteration: {year}')
    
    # Get indices and dates to add to predictions
    dfPred = dfTest[['id', 'eom']].copy()
    dfPred['y_pred'] = y_pred
    
    # Store the predictions in the list
    lPred.append(dfPred)
    
    # Export the predictions
    dfPred.to_csv(f'rf_pred_{year}.csv', index=False)
