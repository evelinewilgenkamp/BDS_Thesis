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

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

###########################################################
### WORKING DIRECTORY
#os.chdir('C:\\Users\\eveli\\Documents\\ThesisData')
os.chdir('D:\\Eveline\\Downloads')


###########################################################
### DATA IMPORT (AND PREP)
dfData = pd.read_csv('jensen2022_data.csv', 
                     sep=',', 
                     index_col=False)

# Drop rows with missing excess returns
dfData.dropna(subset=['ret_exc'], inplace=True)

# Create year and month variable
dfData['eom'] = pd.to_datetime(dfData['eom'], format='%Y-%m-%d')
dfData['year'] = dfData['eom'].dt.year
dfData['month'] = dfData['eom'].dt.month

# Keep relevant part of frame for memory sake
dfData = dfData[(dfData['year'] >= 1960) & (dfData['year'] <= 2020)]

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

lMacro = ['dp', 'ep', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar']

for char in lChars:
    for macro in lMacro:
        dfData[f'{char}X{macro}'] = dfData[char] * dfData[macro]
        
del dfMacro

###########################################################
### FUNCTIONS
def tuneRandomForest(X_train, y_train, X_valid, y_valid):
    # Initialize parameters to tune (similar to Gu et al)
    params_grid = {'max_depth':[1, 2, 3, 4, 5],
                   'max_features':[10, 25, 50, 100, 200, 400]}
    
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
lXvars = ['cowc_gr1a', 'oaccruals_at', 'oaccruals_ni', 'taccruals_at', 
          'taccruals_ni', 'fnl_gr1a', 'ncol_gr1a', 'nfna_gr1a', 'noa_at', 
          'aliq_at', 'at_gr1', 'be_gr1a', 'capx_gr1', 'coa_gr1a', 'col_gr1a', 
          'emp_gr1', 'inv_gr1', 'inv_gr1a', 'lnoa_gr1a', 'mispricing_mgmt', 
          'ncoa_gr1a', 'nncoa_gr1a', 'noa_gr1a', 'ppeinv_gr1a', 'ret_60_12', 
          'sale_gr1', 'seas_2_5na', 'age', 'aliq_mat', 'at_be', 'bidaskhl_21d', 
          'cash_at', 'netdebt_me', 'tangibility', 'beta_60m', 'beta_dimson_21d', 
          'betabab_1260d', 'betadown_252d', 'ivol_capm_21d', 'ivol_capm_252d', 
          'ivol_ff3_21d', 'rmax1_21d', 'rmax5_21d', 'rvol_21d', 'rvol_252d', 
          'turnover_126d', 'zero_trades_126d', 'zero_trades_21d', 'zero_trades_252d', 
          'prc_highprc_252d', 'ret_12_1', 'ret_3_1', 'ret_6_1', 'ret_9_1', 
          'seas_1_1na', 'ocf_at_chg1', 'ret_12_7', 'sale_emp_gr1', 'seas_1_1an', 
          'tax_gr1a', 'dolvol_var_126d', 'ebit_bev', 'ebit_sale', 'intrinsic_value', 
          'ni_be', 'o_score', 'ocf_at', 'ope_be', 'ope_bel1', 'turnover_var_126d', 
          'at_turnover', 'cop_at', 'cop_atl1', 'gp_at', 'gp_atl1', 'mispricing_perf', 
          'op_at', 'op_atl1', 'opex_at', 'qmj_prof', 'qmj_safety', 'sale_bev', 
          'corr_1260d', 'coskew_21d', 'dbnetis_at', 'kz_index', 'lti_gr1a', 
          'pi_nix', 'seas_11_15an', 'seas_11_15na', 'seas_2_5an', 'seas_6_10an', 
          'ami_126d', 'dolvol_126d', 'iskew_capm_21d', 'iskew_ff3_21d', 'ret_1_0', 
          'rmax5_rvol_21d', 'rskew_21d', 'at_me', 'be_me', 'bev_mev', 'chcsho_12m', 
          'debt_me', 'div12m_me', 'ebitda_mev', 'eq_dur', 'eqnpo_12m', 'fcf_me', 
          'ni_me', 'ocf_me', 'sale_me', 'seas_6_10na', 'sic_10', 
          'sic_11', 'sic_12', 'sic_13', 'sic_14', 'sic_15', 'sic_16', 'sic_17', 
          'sic_18', 'sic_20', 'sic_21', 'sic_22', 'sic_23', 'sic_24', 'sic_25', 
          'sic_26', 'sic_27', 'sic_28', 'sic_29', 'sic_30', 'sic_31', 'sic_32', 
          'sic_33', 'sic_34', 'sic_35', 'sic_36', 'sic_37', 'sic_38', 'sic_39', 
          'sic_40', 'sic_41', 'sic_42', 'sic_44', 'sic_45', 'sic_46', 'sic_47', 
          'sic_48', 'sic_49', 'sic_50', 'sic_51', 'sic_52', 'sic_53', 'sic_54', 
          'sic_55', 'sic_56', 'sic_57', 'sic_58', 'sic_59', 'sic_60', 'sic_61', 
          'sic_62', 'sic_63', 'sic_64', 'sic_65', 'sic_66', 'sic_67', 'sic_70', 
          'sic_72', 'sic_73', 'sic_74', 'sic_75', 'sic_76', 'sic_78', 'sic_79', 
          'sic_80', 'sic_81', 'sic_82', 'sic_83', 'sic_87', 'sic_89', 'sic_99', 
          'dp', 'ep', 'b/m', 'ntis', 'tbl', 'tms', 'dfy', 'svar', 'cowc_gr1aXdp', 
          'cowc_gr1aXep', 'cowc_gr1aXb/m', 'cowc_gr1aXntis', 'cowc_gr1aXtbl', 
          'cowc_gr1aXtms', 'cowc_gr1aXdfy', 'cowc_gr1aXsvar', 'oaccruals_atXdp', 
          'oaccruals_atXep', 'oaccruals_atXb/m', 'oaccruals_atXntis', 'oaccruals_atXtbl', 
          'oaccruals_atXtms', 'oaccruals_atXdfy', 'oaccruals_atXsvar', 'oaccruals_niXdp', 'oaccruals_niXep', 'oaccruals_niXb/m', 'oaccruals_niXntis', 'oaccruals_niXtbl', 'oaccruals_niXtms', 'oaccruals_niXdfy', 'oaccruals_niXsvar', 'taccruals_atXdp', 'taccruals_atXep', 'taccruals_atXb/m', 'taccruals_atXntis', 'taccruals_atXtbl', 'taccruals_atXtms', 'taccruals_atXdfy', 'taccruals_atXsvar', 'taccruals_niXdp', 'taccruals_niXep', 'taccruals_niXb/m', 'taccruals_niXntis', 'taccruals_niXtbl', 'taccruals_niXtms', 'taccruals_niXdfy', 'taccruals_niXsvar', 'fnl_gr1aXdp', 'fnl_gr1aXep', 'fnl_gr1aXb/m', 'fnl_gr1aXntis', 'fnl_gr1aXtbl', 'fnl_gr1aXtms', 'fnl_gr1aXdfy', 'fnl_gr1aXsvar', 'ncol_gr1aXdp', 'ncol_gr1aXep', 'ncol_gr1aXb/m', 'ncol_gr1aXntis', 'ncol_gr1aXtbl', 'ncol_gr1aXtms', 'ncol_gr1aXdfy', 'ncol_gr1aXsvar', 'nfna_gr1aXdp', 'nfna_gr1aXep', 'nfna_gr1aXb/m', 'nfna_gr1aXntis', 'nfna_gr1aXtbl', 'nfna_gr1aXtms', 'nfna_gr1aXdfy', 'nfna_gr1aXsvar', 'noa_atXdp', 'noa_atXep', 'noa_atXb/m', 'noa_atXntis', 'noa_atXtbl', 'noa_atXtms', 'noa_atXdfy', 'noa_atXsvar', 'aliq_atXdp', 'aliq_atXep', 'aliq_atXb/m', 'aliq_atXntis', 'aliq_atXtbl', 'aliq_atXtms', 'aliq_atXdfy', 'aliq_atXsvar', 'at_gr1Xdp', 'at_gr1Xep', 'at_gr1Xb/m', 'at_gr1Xntis', 'at_gr1Xtbl', 'at_gr1Xtms', 'at_gr1Xdfy', 'at_gr1Xsvar', 'be_gr1aXdp', 'be_gr1aXep', 'be_gr1aXb/m', 'be_gr1aXntis', 'be_gr1aXtbl', 'be_gr1aXtms', 'be_gr1aXdfy', 'be_gr1aXsvar', 'capx_gr1Xdp', 'capx_gr1Xep', 'capx_gr1Xb/m', 'capx_gr1Xntis', 'capx_gr1Xtbl', 'capx_gr1Xtms', 'capx_gr1Xdfy', 'capx_gr1Xsvar', 'coa_gr1aXdp', 'coa_gr1aXep', 'coa_gr1aXb/m', 'coa_gr1aXntis', 'coa_gr1aXtbl', 'coa_gr1aXtms', 'coa_gr1aXdfy', 'coa_gr1aXsvar', 'col_gr1aXdp', 'col_gr1aXep', 'col_gr1aXb/m', 'col_gr1aXntis', 'col_gr1aXtbl', 'col_gr1aXtms', 'col_gr1aXdfy', 'col_gr1aXsvar', 'emp_gr1Xdp', 'emp_gr1Xep', 'emp_gr1Xb/m', 'emp_gr1Xntis', 'emp_gr1Xtbl', 'emp_gr1Xtms', 'emp_gr1Xdfy', 'emp_gr1Xsvar', 'inv_gr1Xdp', 'inv_gr1Xep', 'inv_gr1Xb/m', 'inv_gr1Xntis', 'inv_gr1Xtbl', 'inv_gr1Xtms', 'inv_gr1Xdfy', 'inv_gr1Xsvar', 'inv_gr1aXdp', 'inv_gr1aXep', 'inv_gr1aXb/m', 'inv_gr1aXntis', 'inv_gr1aXtbl', 'inv_gr1aXtms', 'inv_gr1aXdfy', 'inv_gr1aXsvar', 'lnoa_gr1aXdp', 'lnoa_gr1aXep', 'lnoa_gr1aXb/m', 'lnoa_gr1aXntis', 'lnoa_gr1aXtbl', 'lnoa_gr1aXtms', 'lnoa_gr1aXdfy', 'lnoa_gr1aXsvar', 'mispricing_mgmtXdp', 'mispricing_mgmtXep', 'mispricing_mgmtXb/m', 'mispricing_mgmtXntis', 'mispricing_mgmtXtbl', 'mispricing_mgmtXtms', 'mispricing_mgmtXdfy', 'mispricing_mgmtXsvar', 'ncoa_gr1aXdp', 'ncoa_gr1aXep', 'ncoa_gr1aXb/m', 'ncoa_gr1aXntis', 'ncoa_gr1aXtbl', 'ncoa_gr1aXtms', 'ncoa_gr1aXdfy', 'ncoa_gr1aXsvar', 'nncoa_gr1aXdp', 'nncoa_gr1aXep', 'nncoa_gr1aXb/m', 'nncoa_gr1aXntis', 'nncoa_gr1aXtbl', 'nncoa_gr1aXtms', 'nncoa_gr1aXdfy', 'nncoa_gr1aXsvar', 'noa_gr1aXdp', 'noa_gr1aXep', 'noa_gr1aXb/m', 'noa_gr1aXntis', 'noa_gr1aXtbl', 'noa_gr1aXtms', 'noa_gr1aXdfy', 'noa_gr1aXsvar', 'ppeinv_gr1aXdp', 'ppeinv_gr1aXep', 'ppeinv_gr1aXb/m', 'ppeinv_gr1aXntis', 'ppeinv_gr1aXtbl', 'ppeinv_gr1aXtms', 'ppeinv_gr1aXdfy', 'ppeinv_gr1aXsvar', 'ret_60_12Xdp', 'ret_60_12Xep', 'ret_60_12Xb/m', 'ret_60_12Xntis', 'ret_60_12Xtbl', 'ret_60_12Xtms', 'ret_60_12Xdfy', 'ret_60_12Xsvar', 'sale_gr1Xdp', 'sale_gr1Xep', 'sale_gr1Xb/m', 'sale_gr1Xntis', 'sale_gr1Xtbl', 'sale_gr1Xtms', 'sale_gr1Xdfy', 'sale_gr1Xsvar', 'seas_2_5naXdp', 'seas_2_5naXep', 'seas_2_5naXb/m', 'seas_2_5naXntis', 'seas_2_5naXtbl', 'seas_2_5naXtms', 'seas_2_5naXdfy', 'seas_2_5naXsvar', 'ageXdp', 'ageXep', 'ageXb/m', 'ageXntis', 'ageXtbl', 'ageXtms', 'ageXdfy', 'ageXsvar', 'aliq_matXdp', 'aliq_matXep', 'aliq_matXb/m', 'aliq_matXntis', 'aliq_matXtbl', 'aliq_matXtms', 'aliq_matXdfy', 'aliq_matXsvar', 'at_beXdp', 'at_beXep', 'at_beXb/m', 'at_beXntis', 'at_beXtbl', 'at_beXtms', 'at_beXdfy', 'at_beXsvar', 'bidaskhl_21dXdp', 'bidaskhl_21dXep', 'bidaskhl_21dXb/m', 'bidaskhl_21dXntis', 'bidaskhl_21dXtbl', 'bidaskhl_21dXtms', 'bidaskhl_21dXdfy', 'bidaskhl_21dXsvar', 'cash_atXdp', 'cash_atXep', 'cash_atXb/m', 'cash_atXntis', 'cash_atXtbl', 'cash_atXtms', 'cash_atXdfy', 'cash_atXsvar', 'netdebt_meXdp', 'netdebt_meXep', 'netdebt_meXb/m', 'netdebt_meXntis', 'netdebt_meXtbl', 'netdebt_meXtms', 'netdebt_meXdfy', 'netdebt_meXsvar', 'tangibilityXdp', 'tangibilityXep', 'tangibilityXb/m', 'tangibilityXntis', 'tangibilityXtbl', 'tangibilityXtms', 'tangibilityXdfy', 'tangibilityXsvar', 'beta_60mXdp', 'beta_60mXep', 'beta_60mXb/m', 'beta_60mXntis', 'beta_60mXtbl', 'beta_60mXtms', 'beta_60mXdfy', 'beta_60mXsvar', 'beta_dimson_21dXdp', 'beta_dimson_21dXep', 'beta_dimson_21dXb/m', 'beta_dimson_21dXntis', 'beta_dimson_21dXtbl', 'beta_dimson_21dXtms', 'beta_dimson_21dXdfy', 'beta_dimson_21dXsvar', 'betabab_1260dXdp', 'betabab_1260dXep', 'betabab_1260dXb/m', 'betabab_1260dXntis', 'betabab_1260dXtbl', 'betabab_1260dXtms', 'betabab_1260dXdfy', 'betabab_1260dXsvar', 'betadown_252dXdp', 'betadown_252dXep', 'betadown_252dXb/m', 'betadown_252dXntis', 'betadown_252dXtbl', 'betadown_252dXtms', 'betadown_252dXdfy', 'betadown_252dXsvar', 'ivol_capm_21dXdp', 'ivol_capm_21dXep', 'ivol_capm_21dXb/m', 'ivol_capm_21dXntis', 'ivol_capm_21dXtbl', 'ivol_capm_21dXtms', 'ivol_capm_21dXdfy', 'ivol_capm_21dXsvar', 'ivol_capm_252dXdp', 'ivol_capm_252dXep', 'ivol_capm_252dXb/m', 'ivol_capm_252dXntis', 'ivol_capm_252dXtbl', 'ivol_capm_252dXtms', 'ivol_capm_252dXdfy', 'ivol_capm_252dXsvar', 'ivol_ff3_21dXdp', 'ivol_ff3_21dXep', 'ivol_ff3_21dXb/m', 'ivol_ff3_21dXntis', 'ivol_ff3_21dXtbl', 'ivol_ff3_21dXtms', 'ivol_ff3_21dXdfy', 'ivol_ff3_21dXsvar', 'rmax1_21dXdp', 'rmax1_21dXep', 'rmax1_21dXb/m', 'rmax1_21dXntis', 'rmax1_21dXtbl', 'rmax1_21dXtms', 'rmax1_21dXdfy', 'rmax1_21dXsvar', 'rmax5_21dXdp', 'rmax5_21dXep', 'rmax5_21dXb/m', 'rmax5_21dXntis', 'rmax5_21dXtbl', 'rmax5_21dXtms', 'rmax5_21dXdfy', 'rmax5_21dXsvar', 'rvol_21dXdp', 'rvol_21dXep', 'rvol_21dXb/m', 'rvol_21dXntis', 'rvol_21dXtbl', 'rvol_21dXtms', 'rvol_21dXdfy', 'rvol_21dXsvar', 'rvol_252dXdp', 'rvol_252dXep', 'rvol_252dXb/m', 'rvol_252dXntis', 'rvol_252dXtbl', 'rvol_252dXtms', 'rvol_252dXdfy', 'rvol_252dXsvar', 'turnover_126dXdp', 'turnover_126dXep', 'turnover_126dXb/m', 'turnover_126dXntis', 'turnover_126dXtbl', 'turnover_126dXtms', 'turnover_126dXdfy', 'turnover_126dXsvar', 'zero_trades_126dXdp', 'zero_trades_126dXep', 'zero_trades_126dXb/m', 'zero_trades_126dXntis', 'zero_trades_126dXtbl', 'zero_trades_126dXtms', 'zero_trades_126dXdfy', 'zero_trades_126dXsvar', 'zero_trades_21dXdp', 'zero_trades_21dXep', 'zero_trades_21dXb/m', 'zero_trades_21dXntis', 'zero_trades_21dXtbl', 'zero_trades_21dXtms', 'zero_trades_21dXdfy', 'zero_trades_21dXsvar', 'zero_trades_252dXdp', 'zero_trades_252dXep', 'zero_trades_252dXb/m', 'zero_trades_252dXntis', 'zero_trades_252dXtbl', 'zero_trades_252dXtms', 'zero_trades_252dXdfy', 'zero_trades_252dXsvar', 'prc_highprc_252dXdp', 'prc_highprc_252dXep', 'prc_highprc_252dXb/m', 'prc_highprc_252dXntis', 'prc_highprc_252dXtbl', 'prc_highprc_252dXtms', 'prc_highprc_252dXdfy', 'prc_highprc_252dXsvar', 'ret_12_1Xdp', 'ret_12_1Xep', 'ret_12_1Xb/m', 'ret_12_1Xntis', 'ret_12_1Xtbl', 'ret_12_1Xtms', 'ret_12_1Xdfy', 'ret_12_1Xsvar', 'ret_3_1Xdp', 'ret_3_1Xep', 'ret_3_1Xb/m', 'ret_3_1Xntis', 'ret_3_1Xtbl', 'ret_3_1Xtms', 'ret_3_1Xdfy', 'ret_3_1Xsvar', 'ret_6_1Xdp', 'ret_6_1Xep', 'ret_6_1Xb/m', 'ret_6_1Xntis', 'ret_6_1Xtbl', 'ret_6_1Xtms', 'ret_6_1Xdfy', 'ret_6_1Xsvar', 'ret_9_1Xdp', 'ret_9_1Xep', 'ret_9_1Xb/m', 'ret_9_1Xntis', 'ret_9_1Xtbl', 'ret_9_1Xtms', 'ret_9_1Xdfy', 'ret_9_1Xsvar', 'seas_1_1naXdp', 'seas_1_1naXep', 'seas_1_1naXb/m', 'seas_1_1naXntis', 'seas_1_1naXtbl', 'seas_1_1naXtms', 'seas_1_1naXdfy', 'seas_1_1naXsvar', 'ocf_at_chg1Xdp', 'ocf_at_chg1Xep', 'ocf_at_chg1Xb/m', 'ocf_at_chg1Xntis', 'ocf_at_chg1Xtbl', 'ocf_at_chg1Xtms', 'ocf_at_chg1Xdfy', 'ocf_at_chg1Xsvar', 'ret_12_7Xdp', 'ret_12_7Xep', 'ret_12_7Xb/m', 'ret_12_7Xntis', 'ret_12_7Xtbl', 'ret_12_7Xtms', 'ret_12_7Xdfy', 'ret_12_7Xsvar', 'sale_emp_gr1Xdp', 'sale_emp_gr1Xep', 'sale_emp_gr1Xb/m', 'sale_emp_gr1Xntis', 'sale_emp_gr1Xtbl', 'sale_emp_gr1Xtms', 'sale_emp_gr1Xdfy', 'sale_emp_gr1Xsvar', 'seas_1_1anXdp', 'seas_1_1anXep', 'seas_1_1anXb/m', 'seas_1_1anXntis', 'seas_1_1anXtbl', 'seas_1_1anXtms', 'seas_1_1anXdfy', 'seas_1_1anXsvar', 'tax_gr1aXdp', 'tax_gr1aXep', 'tax_gr1aXb/m', 'tax_gr1aXntis', 'tax_gr1aXtbl', 'tax_gr1aXtms', 'tax_gr1aXdfy', 'tax_gr1aXsvar', 'dolvol_var_126dXdp', 'dolvol_var_126dXep', 'dolvol_var_126dXb/m', 'dolvol_var_126dXntis', 'dolvol_var_126dXtbl', 'dolvol_var_126dXtms', 'dolvol_var_126dXdfy', 'dolvol_var_126dXsvar', 'ebit_bevXdp', 'ebit_bevXep', 'ebit_bevXb/m', 'ebit_bevXntis', 'ebit_bevXtbl', 'ebit_bevXtms', 'ebit_bevXdfy', 'ebit_bevXsvar', 'ebit_saleXdp', 'ebit_saleXep', 'ebit_saleXb/m', 'ebit_saleXntis', 'ebit_saleXtbl', 'ebit_saleXtms', 'ebit_saleXdfy', 'ebit_saleXsvar', 'intrinsic_valueXdp', 'intrinsic_valueXep', 'intrinsic_valueXb/m', 'intrinsic_valueXntis', 'intrinsic_valueXtbl', 'intrinsic_valueXtms', 'intrinsic_valueXdfy', 'intrinsic_valueXsvar', 'ni_beXdp', 'ni_beXep', 'ni_beXb/m', 'ni_beXntis', 'ni_beXtbl', 'ni_beXtms', 'ni_beXdfy', 'ni_beXsvar', 'o_scoreXdp', 'o_scoreXep', 'o_scoreXb/m', 'o_scoreXntis', 'o_scoreXtbl', 'o_scoreXtms', 'o_scoreXdfy', 'o_scoreXsvar', 'ocf_atXdp', 'ocf_atXep', 'ocf_atXb/m', 'ocf_atXntis', 'ocf_atXtbl', 'ocf_atXtms', 'ocf_atXdfy', 'ocf_atXsvar', 'ope_beXdp', 'ope_beXep', 'ope_beXb/m', 'ope_beXntis', 'ope_beXtbl', 'ope_beXtms', 'ope_beXdfy', 'ope_beXsvar', 'ope_bel1Xdp', 'ope_bel1Xep', 'ope_bel1Xb/m', 'ope_bel1Xntis', 'ope_bel1Xtbl', 'ope_bel1Xtms', 'ope_bel1Xdfy', 'ope_bel1Xsvar', 'turnover_var_126dXdp', 'turnover_var_126dXep', 'turnover_var_126dXb/m', 'turnover_var_126dXntis', 'turnover_var_126dXtbl', 'turnover_var_126dXtms', 'turnover_var_126dXdfy', 'turnover_var_126dXsvar', 'at_turnoverXdp', 'at_turnoverXep', 'at_turnoverXb/m', 'at_turnoverXntis', 'at_turnoverXtbl', 'at_turnoverXtms', 'at_turnoverXdfy', 'at_turnoverXsvar', 'cop_atXdp', 'cop_atXep', 'cop_atXb/m', 'cop_atXntis', 'cop_atXtbl', 'cop_atXtms', 'cop_atXdfy', 'cop_atXsvar', 'cop_atl1Xdp', 'cop_atl1Xep', 'cop_atl1Xb/m', 'cop_atl1Xntis', 'cop_atl1Xtbl', 'cop_atl1Xtms', 'cop_atl1Xdfy', 'cop_atl1Xsvar', 'gp_atXdp', 'gp_atXep', 'gp_atXb/m', 'gp_atXntis', 'gp_atXtbl', 'gp_atXtms', 'gp_atXdfy', 'gp_atXsvar', 'gp_atl1Xdp', 'gp_atl1Xep', 'gp_atl1Xb/m', 'gp_atl1Xntis', 'gp_atl1Xtbl', 'gp_atl1Xtms', 'gp_atl1Xdfy', 'gp_atl1Xsvar', 'mispricing_perfXdp', 'mispricing_perfXep', 'mispricing_perfXb/m', 'mispricing_perfXntis', 'mispricing_perfXtbl', 'mispricing_perfXtms', 'mispricing_perfXdfy', 'mispricing_perfXsvar', 'op_atXdp', 'op_atXep', 'op_atXb/m', 'op_atXntis', 'op_atXtbl', 'op_atXtms', 'op_atXdfy', 'op_atXsvar', 'op_atl1Xdp', 'op_atl1Xep', 'op_atl1Xb/m', 'op_atl1Xntis', 'op_atl1Xtbl', 'op_atl1Xtms', 'op_atl1Xdfy', 'op_atl1Xsvar', 'opex_atXdp', 'opex_atXep', 'opex_atXb/m', 'opex_atXntis', 'opex_atXtbl', 'opex_atXtms', 'opex_atXdfy', 'opex_atXsvar', 'qmj_profXdp', 'qmj_profXep', 'qmj_profXb/m', 'qmj_profXntis', 'qmj_profXtbl', 'qmj_profXtms', 'qmj_profXdfy', 'qmj_profXsvar', 'qmj_safetyXdp', 'qmj_safetyXep', 'qmj_safetyXb/m', 'qmj_safetyXntis', 'qmj_safetyXtbl', 'qmj_safetyXtms', 'qmj_safetyXdfy', 'qmj_safetyXsvar', 'sale_bevXdp', 'sale_bevXep', 'sale_bevXb/m', 'sale_bevXntis', 'sale_bevXtbl', 'sale_bevXtms', 'sale_bevXdfy', 'sale_bevXsvar', 'corr_1260dXdp', 'corr_1260dXep', 'corr_1260dXb/m', 'corr_1260dXntis', 'corr_1260dXtbl', 'corr_1260dXtms', 'corr_1260dXdfy', 'corr_1260dXsvar', 'coskew_21dXdp', 'coskew_21dXep', 'coskew_21dXb/m', 'coskew_21dXntis', 'coskew_21dXtbl', 'coskew_21dXtms', 'coskew_21dXdfy', 'coskew_21dXsvar', 'dbnetis_atXdp', 'dbnetis_atXep', 'dbnetis_atXb/m', 'dbnetis_atXntis', 'dbnetis_atXtbl', 'dbnetis_atXtms', 'dbnetis_atXdfy', 'dbnetis_atXsvar', 'kz_indexXdp', 'kz_indexXep', 'kz_indexXb/m', 'kz_indexXntis', 'kz_indexXtbl', 'kz_indexXtms', 'kz_indexXdfy', 'kz_indexXsvar', 'lti_gr1aXdp', 'lti_gr1aXep', 'lti_gr1aXb/m', 'lti_gr1aXntis', 'lti_gr1aXtbl', 'lti_gr1aXtms', 'lti_gr1aXdfy', 'lti_gr1aXsvar', 'pi_nixXdp', 'pi_nixXep', 'pi_nixXb/m', 'pi_nixXntis', 'pi_nixXtbl', 'pi_nixXtms', 'pi_nixXdfy', 'pi_nixXsvar', 'seas_11_15anXdp', 'seas_11_15anXep', 'seas_11_15anXb/m', 'seas_11_15anXntis', 'seas_11_15anXtbl', 'seas_11_15anXtms', 'seas_11_15anXdfy', 'seas_11_15anXsvar', 'seas_11_15naXdp', 'seas_11_15naXep', 'seas_11_15naXb/m', 'seas_11_15naXntis', 'seas_11_15naXtbl', 'seas_11_15naXtms', 'seas_11_15naXdfy', 'seas_11_15naXsvar', 'seas_2_5anXdp', 'seas_2_5anXep', 'seas_2_5anXb/m', 'seas_2_5anXntis', 'seas_2_5anXtbl', 'seas_2_5anXtms', 'seas_2_5anXdfy', 'seas_2_5anXsvar', 'seas_6_10anXdp', 'seas_6_10anXep', 'seas_6_10anXb/m', 'seas_6_10anXntis', 'seas_6_10anXtbl', 'seas_6_10anXtms', 'seas_6_10anXdfy', 'seas_6_10anXsvar', 'ami_126dXdp', 'ami_126dXep', 'ami_126dXb/m', 'ami_126dXntis', 'ami_126dXtbl', 'ami_126dXtms', 'ami_126dXdfy', 'ami_126dXsvar', 'dolvol_126dXdp', 'dolvol_126dXep', 'dolvol_126dXb/m', 'dolvol_126dXntis', 'dolvol_126dXtbl', 'dolvol_126dXtms', 'dolvol_126dXdfy', 'dolvol_126dXsvar', 'market_equityXdp', 'market_equityXep', 'market_equityXb/m', 'market_equityXntis', 'market_equityXtbl', 'market_equityXtms', 'market_equityXdfy', 'market_equityXsvar', 'prcXdp', 'prcXep', 'prcXb/m', 'prcXntis', 'prcXtbl', 'prcXtms', 'prcXdfy', 'prcXsvar', 'iskew_capm_21dXdp', 'iskew_capm_21dXep', 'iskew_capm_21dXb/m', 'iskew_capm_21dXntis', 'iskew_capm_21dXtbl', 'iskew_capm_21dXtms', 'iskew_capm_21dXdfy', 'iskew_capm_21dXsvar', 'iskew_ff3_21dXdp', 'iskew_ff3_21dXep', 'iskew_ff3_21dXb/m', 'iskew_ff3_21dXntis', 'iskew_ff3_21dXtbl', 'iskew_ff3_21dXtms', 'iskew_ff3_21dXdfy', 'iskew_ff3_21dXsvar', 'ret_1_0Xdp', 'ret_1_0Xep', 'ret_1_0Xb/m', 'ret_1_0Xntis', 'ret_1_0Xtbl', 'ret_1_0Xtms', 'ret_1_0Xdfy', 'ret_1_0Xsvar', 'rmax5_rvol_21dXdp', 'rmax5_rvol_21dXep', 'rmax5_rvol_21dXb/m', 'rmax5_rvol_21dXntis', 'rmax5_rvol_21dXtbl', 'rmax5_rvol_21dXtms', 'rmax5_rvol_21dXdfy', 'rmax5_rvol_21dXsvar', 'rskew_21dXdp', 'rskew_21dXep', 'rskew_21dXb/m', 'rskew_21dXntis', 'rskew_21dXtbl', 'rskew_21dXtms', 'rskew_21dXdfy', 'rskew_21dXsvar', 'at_meXdp', 'at_meXep', 'at_meXb/m', 'at_meXntis', 'at_meXtbl', 'at_meXtms', 'at_meXdfy', 'at_meXsvar', 'be_meXdp', 'be_meXep', 'be_meXb/m', 'be_meXntis', 'be_meXtbl', 'be_meXtms', 'be_meXdfy', 'be_meXsvar', 'bev_mevXdp', 'bev_mevXep', 'bev_mevXb/m', 'bev_mevXntis', 'bev_mevXtbl', 'bev_mevXtms', 'bev_mevXdfy', 'bev_mevXsvar', 'chcsho_12mXdp', 'chcsho_12mXep', 'chcsho_12mXb/m', 'chcsho_12mXntis', 'chcsho_12mXtbl', 'chcsho_12mXtms', 'chcsho_12mXdfy', 'chcsho_12mXsvar', 'debt_meXdp', 'debt_meXep', 'debt_meXb/m', 'debt_meXntis', 'debt_meXtbl', 'debt_meXtms', 'debt_meXdfy', 'debt_meXsvar', 'div12m_meXdp', 'div12m_meXep', 'div12m_meXb/m', 'div12m_meXntis', 'div12m_meXtbl', 'div12m_meXtms', 'div12m_meXdfy', 'div12m_meXsvar', 'ebitda_mevXdp', 'ebitda_mevXep', 'ebitda_mevXb/m', 'ebitda_mevXntis', 'ebitda_mevXtbl', 'ebitda_mevXtms', 'ebitda_mevXdfy', 'ebitda_mevXsvar', 'eq_durXdp', 'eq_durXep', 'eq_durXb/m', 'eq_durXntis', 'eq_durXtbl', 'eq_durXtms', 'eq_durXdfy', 'eq_durXsvar', 'eqnpo_12mXdp', 'eqnpo_12mXep', 'eqnpo_12mXb/m', 'eqnpo_12mXntis', 'eqnpo_12mXtbl', 'eqnpo_12mXtms', 'eqnpo_12mXdfy', 'eqnpo_12mXsvar', 'fcf_meXdp', 'fcf_meXep', 'fcf_meXb/m', 'fcf_meXntis', 'fcf_meXtbl', 'fcf_meXtms', 'fcf_meXdfy', 'fcf_meXsvar', 'ni_meXdp', 'ni_meXep', 'ni_meXb/m', 'ni_meXntis', 'ni_meXtbl', 'ni_meXtms', 'ni_meXdfy', 'ni_meXsvar', 'ocf_meXdp', 'ocf_meXep', 'ocf_meXb/m', 'ocf_meXntis', 'ocf_meXtbl', 'ocf_meXtms', 'ocf_meXdfy', 'ocf_meXsvar', 'sale_meXdp', 'sale_meXep', 'sale_meXb/m', 'sale_meXntis', 'sale_meXtbl', 'sale_meXtms', 'sale_meXdfy', 'sale_meXsvar', 'seas_6_10naXdp', 'seas_6_10naXep', 'seas_6_10naXb/m', 'seas_6_10naXntis', 'seas_6_10naXtbl', 'seas_6_10naXtms', 'seas_6_10naXdfy', 'seas_6_10naXsvar']

# Set initial train, valid, test split
train_start = 1960
train_end = 1978
valid_start = 1979
valid_end = 1990
test_start = 1991
test_end = 2020

# List to store the predictions and best params in
lParams = []
lPred = []

# Expanding window loop
for year in range(test_start, test_end+1):
    print(f'Current prediction iteration: {year}')
    
    # Prepare train, validation, and test tests
    dfTrain = dfData[(dfData['year'] >= train_start) & (dfData['year'] <= train_end)]
    dfValid = dfData[(dfData['year'] >= valid_start) & (dfData['year'] <= valid_end)]
    dfTest = dfData[dfData['year'] == test_start]
    
    X_train = dfTrain[lXvars]
    y_train = dfTrain['ret_exc']
    X_valid = dfValid[lXvars]
    y_valid = dfValid['ret_exc']
    X_test = dfTest[lXvars]
    y_test = dfTest['ret_exc']
    
    del dfTrain, dfValid, dfTest
    
    print(f'Tuning started for iteration: {year}')
    
    # Find the best parameters for the model refit
    best_params = tuneRandomForest(X_train, y_train, X_valid, y_valid)
    lParams.append(best_params)
    
    print(f'Tuning finished for iteration: {year}')
    print(f'Best params for year {year} are {best_params}')
    
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
