# -*- coding: utf-8 -*-
"""
RandomForestPredictions.py

Purpose:
    Predict returns of NYSE, AMEX, NASDAQ using random forest
    Train: 1991-2000, valid: 2001-2002, test:2003-2020, rolling window

Version:
    Final
    
Date:
    27-06-2023

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

###########################################################
### WORKING DIRECTORY
os.chdir('C:\\Users\\eveli\\Documents\\ThesisData')
#os.chdir('D:\\Eveline\\Downloads')


###########################################################
### DATA IMPORT (AND PREP)
#dfData = pd.read_csv('fullsample_data.csv', sep=',', index_col=False)
dfData = pd.read_csv('fullsample_data_2.csv', sep=',', index_col=False)
#dfData = pd.read_csv('jensen2022_data.csv', sep=',', index_col=False)
#dfData = pd.read_csv('largecap_data.csv', sep=',', index_col=False)
#dfData = pd.read_csv('nyse_data.csv', sep=',', index_col=False)
#dfData = pd.read_csv('latesample_data.csv', sep=',', index_col=False)

# Markit
dfShort = pd.read_csv('markit.csv', sep=',', index_col=False)
dfShort['mdate'] = pd.to_datetime(dfShort['mdate'], format='%Y%m')
dfShort['year'] = dfShort['mdate'].dt.year
dfShort['month'] = dfShort['mdate'].dt.month
shortyears = set(dfShort['year'].values.tolist()) #2002-2021

# Drop rows with missing excess returns
dfData.dropna(subset=['ret_exc'], inplace=True)

# Create year and month variable
dfData['eom'] = pd.to_datetime(dfData['eom'], format='%Y-%m-%d')
dfData['year'] = dfData['eom'].dt.year
dfData['month'] = dfData['eom'].dt.month

# One hot encoding of sic codes
dfData['sic'] = dfData.groupby(['id']).ffill()['sic']
dfData['sic'] = dfData.groupby(['id']).bfill()['sic']
dfData.dropna(subset=['sic'], inplace=True)
dfData['sic'] = dfData['sic'].astype(str).str[:2].astype(int)
dfData = pd.get_dummies(dfData, columns=['sic'], dtype=int)

# Welch & Goyal (2008) macro data pp1457 for definitions
dfMacro = pd.read_csv('WelchGoyalData.csv', sep=',', index_col=False)
dfMacro['yyyymm'] = pd.to_datetime(dfMacro['yyyymm'], format='%Y%m')
dfMacro['year'] = dfMacro['yyyymm'].dt.year
dfMacro['month'] = dfMacro['yyyymm'].dt.month
dfMacro = dfMacro[(dfMacro['year'] >= 1960) & (dfMacro['year'] <= 2020)]
# Dividend-price ratio
dfMacro['Index'] = dfMacro['Index'].replace(',','', regex=True)
dfMacro['Index'] = dfMacro['Index'].astype(float)
dfMacro['dp'] = np.log(dfMacro['D12']) - np.log(dfMacro['Index'])
# Earnings-price ratio
dfMacro['ep'] = np.log(dfMacro['E12']) - np.log(dfMacro['Index'])
# Term spread
dfMacro['tms'] = dfMacro['lty'] - dfMacro['tbl']
# Default spread
dfMacro['dfy'] = dfMacro['BAA'] - dfMacro['AAA']
# List to keep
lKeep = ['dp', 'ep', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar', 'year', 'month']
dfMacro = dfMacro[lKeep]
# List of macro variables
lMacro = ['dp', 'ep', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar']

# Merges
dfData = pd.merge(dfData, dfMacro, 
                  how='left',
                  left_on=['year','month'], 
                  right_on = ['year','month'])

# Interaction variables
# Characteristics
lChars = ['cowc_gr1a', 'oaccruals_at', 'oaccruals_ni', 
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


for char in lChars:
    for macro in lMacro:
        dfData[f'{char}X{macro}'] = dfData[char] * dfData[macro]
        
dfData.drop(lMacro, axis=1, inplace=True)

lColumns = dfData.columns.tolist()
lColumns = lColumns[32:]
lColumns.insert(94, 'market_equity')
lColumns.insert(95, 'prc')
lColumns.remove('year')
lColumns.remove('month')
        
del dfMacro

# Defragment frame
dfData = dfData.copy()

###########################################################
### FUNCTIONS
def tuneRandomForest(X_train, y_train, X_valid, y_valid):
    # Initialize parameters to tune (similar to Gu et al)
    params_grid = {'max_depth':[1, 2, 3, 4, 5],
                   'max_features':[10, 25, 50, 100, 200]}
    
    # Initializations the random forest regressor
    rfg = RandomForestRegressor(n_estimators=300,
                                random_state=69,
                                n_jobs=6)
    
    best_score = np.NINF
    best_params = None
    
    # Loop through the grid and find best combinations
    for max_depth in params_grid['max_depth']:
        for max_features in params_grid['max_features']:
            # Set new parameters
            rfg.set_params(max_depth=max_depth,
                           max_features=max_features)
            
            # Fit on the training data
            t0 = time.time()
            rfg.fit(X=X_train, y=y_train)
            t1 = time.time()
            total = t1-t0
            
            # Evaluate performance on validation set
            score = rfg.score(X=X_valid, y=y_valid)
            print(f'model maxdepth {max_depth} and feat {max_features} took {total} seconds with score {score}')
            
            
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
# Set initial train, valid, test split
train_start = 1999
train_end = 2008
valid_start = 2009
valid_end = 2010
test_start = 2011
test_end = 2020

# List to store the predictions and best params in
lParams = []
lPred = []

# Rolling window loop
for year in range(test_start, test_end+1):
    print(f'Current prediction iteration: {year}')
    print(f'Training parameters: train start {train_start}, train end {train_end}, valid start {valid_start}, valid end {valid_end}, test set {year}')
    
    # Prepare train, validation, and test tests
    dfTest = dfData[dfData['year'] == year]
    
    X_train = dfData[(dfData['year'] >= train_start) & (dfData['year'] <= train_end)][lColumns]
    y_train = dfData[(dfData['year'] >= train_start) & (dfData['year'] <= train_end)]['ret_exc']
    X_valid = dfData[(dfData['year'] >= valid_start) & (dfData['year'] <= valid_end)][lColumns]
    y_valid = dfData[(dfData['year'] >= valid_start) & (dfData['year'] <= valid_end)]['ret_exc']
    X_test = dfTest[lColumns]
    y_test = dfTest['ret_exc']
    
    print(f'Tuning started for iteration: {year}')
    
    # Find the best parameters for the model refit
    best_params = tuneRandomForest(X_train, y_train, X_valid, y_valid)
    best_params['year'] = year
    lParams.append(best_params)
    
    print(f'Tuning finished for iteration: {year}')
    print(f'Best params for year {year} are {best_params}')
    
    print(f'Predictions started for iteration: {year}')
    
    # Fit the model for the prediction year at hand
    y_pred = predictRandomForest(X_train, y_train, X_test, y_test, best_params)
    
    # Get indices and dates to add to predictions
    dfPred = dfTest[['id', 'eom']].copy()
    dfPred['y_pred'] = y_pred
    
    # Store the predictions in the list
    lPred.append(dfPred)
    
    # Export the predictions
    dfPred.to_csv(f'rf_pred_{year}.csv', index=False)
    
    # Set new rolling window
    train_start = train_start + 1
    train_end = train_end + 1
    valid_start = valid_start + 1
    valid_end = valid_end + 1


# backtesting
r2_test = r2_score(y_test, y_pred)
dfPred[dfPred['y_pred'] == dfPred['y_pred'].quantile(0.9)]
dfPred[dfPred['y_pred'] == dfPred['y_pred'].quantile(0.1)]
