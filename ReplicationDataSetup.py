# -*- coding: utf-8 -*-
"""
ReplicationDataSetup.py

Purpose:
    Prepare all data necessary to replicate the paper by Jensen, Kelly,
    Malamud, & Pedersen (2022)

Version:
    3. Final version (done)
    
Date:
    22-03-2023

Author:
    Eveline Wilgenkamp
"""

###########################################################
### IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import date, timedelta, datetime
from dateutil.parser import parse
from sklearn.preprocessing import minmax_scale

###########################################################
### WORKING DIRECTORY
os.chdir('C:\\Users\\eveli\\Documents\\ThesisData')

###########################################################
### IMPORTANT NOTE
# To prepare the dataset used by Jensen et al. (2022), we can use the 
# replicability code accessable at https://github.com/bkelly-lab/ReplicationCrisis
# SAS files extracted, ran in SAS Online through WRDS portal

###########################################################
### DATA IMPORT
# Characteristics frame is big (approx 15GB), need at least 32GB ram to run!

# Memory saving: set categorical vars
dTypes = {'size_grp':'category'}

# Memory saving: drop unneccesary columns
lDrop = ['obs_main', 'exch_main', 'primary_sec', 'excntry', 'curcd', 'fx',
         'common', 'comp_tpci', 'crsp_shrcd', 'comp_exchg', 'crsp_exchcd',
         'prc_local', 'ret_local',]
          
dfChar = pd.read_csv('usa_char.csv', 
                     sep=',', 
                     index_col=False, 
                     low_memory=True,
                     dtype = dTypes,
                     usecols = lambda x: x not in lDrop)

###########################################################
### DATA CLEANING AND PREPARATION
# Marketcap above 50th percentile of NYSE
dfChar = dfChar[(dfChar['size_grp'] == 'large') | (dfChar['size_grp'] == 'mega')]

# Select 115 stock characteristics from Jensen et al. 2022
lKeep = ['id', 'date', 'eom', 'source_crsp', 'size_grp', 'gvkey', 'iid',
         'permno', 'permco', 'adjfct', 'shares', 'me', 'me_lag1', 'gics',
         'sic', 'naics', 'ff49', 'me_company', 'prc', 'dolvol', 'ret',
         'ret_exc', 'ret_lag_dif', 'ret_exc_lead1m', 'market_equity', 
         'enterprise_value', 'book_equity', 'sales', 'net_income', 'bidask',
         'prc_high', 'prc_low', 'tvol', 'cowc_gr1a', 'oaccruals_at',
         'oaccruals_ni', 'taccruals_at', 'taccruals_ni', 'fnl_gr1a',
         'ncol_gr1a', 'nfna_gr1a', 'noa_at', 'aliq_at', 'at_gr1', 'be_gr1a',
         'capx_gr1', 'coa_gr1a', 'col_gr1a', 'emp_gr1', 'inv_gr1', 'inv_gr1a',
         'lnoa_gr1a', 'mispricing_mgmt', 'ncoa_gr1a', 'nncoa_gr1a', 'noa_gr1a',
         'ppeinv_gr1a', 'ret_60_12', 'sale_gr1', 'seas_2_5na', 'age', 
         'aliq_mat', 'at_be', 'bidaskhl_21d', 'cash_at', 'netdebt_me', 
         'tangibility', 'beta_60m', 'beta_dimson_21d', 'betabab_1260d',
         'betadown_252d', 'ivol_capm_21d', 'ivol_capm_252d', 'ivol_ff3_21d',
         'rmax1_21d', 'rmax5_21d', 'rvol_21d', 'rvol_252d', 'turnover_126d', 
         'zero_trades_126d', 'zero_trades_21d', 'zero_trades_252d', 
         'prc_highprc_252d', 'ret_12_1', 'ret_3_1', 'ret_6_1', 'ret_9_1',
         'seas_1_1na', 'ocf_at_chg1', 'ret_12_7', 'sale_emp_gr1', 'seas_1_1an',
         'tax_gr1a', 'dolvol_var_126d', 'ebit_bev', 'ebit_sale', 
         'intrinsic_value', 'ni_be', 'o_score', 'ocf_at', 'ope_be', 'ope_bel1',
         'turnover_var_126d', 'at_turnover', 'cop_at', 'cop_atl1', 'gp_at',
         'gp_atl1', 'mispricing_perf', 'op_at', 'op_atl1', 'opex_at', 
         'qmj_prof', 'qmj_safety', 'sale_bev', 'corr_1260d', 'coskew_21d',
         'dbnetis_at', 'kz_index', 'lti_gr1a', 'pi_nix', 'seas_11_15an',
         'seas_11_15na', 'seas_2_5an', 'seas_6_10an', 'ami_126d', 
         'dolvol_126d', 'iskew_capm_21d', 'iskew_ff3_21d', 'ret_1_0', 
         'rmax5_rvol_21d', 'rskew_21d', 'at_me', 'be_me', 'bev_mev', 
         'chcsho_12m', 'debt_me', 'div12m_me', 'ebitda_mev', 'eq_dur', 
         'eqnpo_12m', 'fcf_me', 'ni_me', 'ocf_me', 'sale_me', 'seas_6_10na']

dfChar = dfChar[lKeep]
dfChar = dfChar.loc[:, ~dfChar.columns.duplicated()]

# List of features
lFeatures = ['cowc_gr1a', 'oaccruals_at', 'oaccruals_ni', 'taccruals_at', 
             'taccruals_ni', 'fnl_gr1a', 'ncol_gr1a', 'nfna_gr1a', 'noa_at', 
             'aliq_at', 'at_gr1', 'be_gr1a', 'capx_gr1', 'coa_gr1a', 
             'col_gr1a', 'emp_gr1', 'inv_gr1', 'inv_gr1a', 'lnoa_gr1a', 
             'mispricing_mgmt', 'ncoa_gr1a', 'nncoa_gr1a', 'noa_gr1a',
             'ppeinv_gr1a', 'ret_60_12', 'sale_gr1', 'seas_2_5na', 'age', 
             'aliq_mat', 'at_be', 'bidaskhl_21d', 'cash_at', 'netdebt_me', 
             'tangibility', 'beta_60m', 'beta_dimson_21d', 'betabab_1260d',
             'betadown_252d', 'ivol_capm_21d', 'ivol_capm_252d', 
             'ivol_ff3_21d', 'rmax1_21d', 'rmax5_21d', 'rvol_21d', 'rvol_252d',
             'turnover_126d', 'zero_trades_126d', 'zero_trades_21d', 
             'zero_trades_252d', 'prc_highprc_252d', 'ret_12_1', 'ret_3_1', 
             'ret_6_1', 'ret_9_1', 'seas_1_1na', 'ocf_at_chg1', 'ret_12_7', 
             'sale_emp_gr1', 'seas_1_1an', 'tax_gr1a', 'dolvol_var_126d', 
             'ebit_bev', 'ebit_sale', 'intrinsic_value', 'ni_be', 'o_score', 
             'ocf_at', 'ope_be', 'ope_bel1', 'turnover_var_126d', 
             'at_turnover', 'cop_at', 'cop_atl1', 'gp_at', 'gp_atl1', 
             'mispricing_perf', 'op_at', 'op_atl1', 'opex_at', 
             'qmj_prof', 'qmj_safety', 'sale_bev', 'corr_1260d', 'coskew_21d',
             'dbnetis_at', 'kz_index', 'lti_gr1a', 'pi_nix', 'seas_11_15an',
             'seas_11_15na', 'seas_2_5an', 'seas_6_10an', 'ami_126d', 
             'dolvol_126d', 'market_equity', 'prc', 'iskew_capm_21d',
             'iskew_ff3_21d', 'ret_1_0', 'rmax5_rvol_21d', 'rskew_21d', 
             'at_me', 'be_me', 'bev_mev', 'chcsho_12m', 'debt_me', 'div12m_me', 
             'ebitda_mev', 'eq_dur', 'eqnpo_12m', 'fcf_me', 'ni_me', 'ocf_me',
             'sale_me', 'seas_6_10na']

# Lag features (prevent lookahead bias)
for feature in lFeatures:
    dfChar[feature] = dfChar.groupby('id')[feature].shift(1)

# Drop missing ME at beginning of month (me_lag1)
dfChar.dropna(subset=['me_lag1'], inplace=True)

# Drop rows with > 57 missing features
dfChar['missing_count'] = dfChar[lFeatures].isnull().sum(axis=1)
dfChar = dfChar[dfChar['missing_count'] < 58]

# Sample time: 1952-2020
dfChar['eom'] = pd.to_datetime(dfChar['eom'], format='%Y%m%d')
dfChar['year'] = dfChar['eom'].dt.year
dfChar = dfChar[dfChar['year'] <= 2020]
dfChar = dfChar[dfChar['year'] >= 1952]

# Compute cross-sectional rank each month
for feature in lFeatures:
    dfChar[feature] = dfChar.groupby(['eom'])[feature].rank()
    
# Map to [0,1] interval
for feature in lFeatures:
    dfChar[feature] = dfChar.groupby(['eom'])[feature].transform(
        lambda x: minmax_scale(x.astype(float)))

# Set missing values to 0.5
dfChar[lFeatures] = dfChar[lFeatures].fillna(0.5)

# Drop helper columns
dfChar = dfChar.drop(columns=['year', 'missing_count'])

# Export final dataset
dfChar.to_csv('jensen2022_data.csv', index=False)

###########################################################
### Downloaded anomalies for replication check
# JKP factors (US, capped value-weighted as in Jensen et al. 2022)
dfSignal = pd.read_csv('JKP_Factors.csv',
                       sep=',',
                       index_col=False)

# Drop redundant columns
dfSignal.drop(columns=['location', 'freq', 'weighting'], inplace=True)

# Make columns of factors
dfSignal = dfSignal.pivot(index='date', columns='name')
