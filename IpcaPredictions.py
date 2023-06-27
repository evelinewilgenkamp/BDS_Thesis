# -*- coding: utf-8 -*-
"""
IpcaPredictions.py

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from ipca import InstrumentedPCA


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


###########################################################
### WORKING DIRECTORY
os.chdir('C:\\Users\\eveli\\Documents\\ThesisData')
#os.chdir('D:\\Eveline\\Downloads')


###########################################################
### DATA IMPORT (AND PREP)
dfData = pd.read_csv('fullsample_data_2.csv', sep=',', index_col=False)

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
# Lag by 1 month (lookahead bias)
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
lColumns = ['id', 'eom'] + lColumns[32:]
lColumns.remove('year')
lColumns.remove('month')
        
del dfMacro

# Defragment frame
dfData = dfData.copy()

##################
# Set initial train, valid, test split
train_start = 2001
train_end = 2010
valid_start = 2011
valid_end = 2012
test_start = 2013
test_end = 2020

### NOTES ON THE NEXT SECTION
# (1) Multi-index blows up memory usage of main frame
#     We must apply it to the smaller subframes to save memory
# (2) For some of the subframes, the SIC dummies contain only 0s
#     This means that the data matrix is no longer invertible -> no regression
#     We must fix this by dropping the redundant dummies from all frames

# Prepare train, validation, and test tests
dfTest = dfData[dfData['year'] == test_start]


X_train = dfData[(dfData['year'] >= train_start) & (dfData['year'] <= train_end)][lColumns]
X_train = X_train.set_index(['id', 'eom'])
# Check if training data has only zeros (SIC dummies)
lAllZero = X_train.columns[(X_train == 0).all()].to_list()
X_train.drop(lAllZero, axis=1, inplace=True)


# Drop highly correlated features
correlation = pd.DataFrame(np.corrcoef(X_train.values, rowvar=False), columns=X_train.columns).abs()
correlation.index = correlation.columns
correlation = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
stacked_corr = correlation.unstack()

lCorrelated = []
for index, value in stacked_corr.items():
    if value > 0.95:
        lCorrelated.append(index)

to_drop = [column for column in correlation.columns if any(correlation[column] > 0.95)]
X_train.drop(to_drop, axis=1, inplace=True)

y_train = dfData[(dfData['year'] >= train_start) & (dfData['year'] <= train_end)][['id', 'eom', 'ret_exc']]
y_train = y_train.set_index(['id', 'eom'])['ret_exc']



X_valid = dfData[(dfData['year'] >= valid_start) & (dfData['year'] <= valid_end)][lColumns]
X_valid = X_valid.set_index(['id', 'eom'])



y_valid = dfData[(dfData['year'] >= valid_start) & (dfData['year'] <= valid_end)][['id', 'eom', 'ret_exc']]
y_valid = y_valid.set_index(['id', 'eom'])['ret_exc']


X_test = dfTest[lColumns]
X_test = X_test.set_index(['id', 'eom'])


y_test = dfTest[['id', 'eom', 'ret_exc']]
y_test = y_test.set_index(['id', 'eom'])['ret_exc']

ipcareg = InstrumentedPCA(n_factors=5, intercept=False, n_jobs=6)
result = ipcareg.fit(X=X_train, y=y_train, indices=['id', 'eom'])































