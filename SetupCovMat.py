# -*- coding: utf-8 -*-
"""
SetupCovMat.py

Purpose:
    Set up the dtaa for the (co)variance matrix needed for the portfolio 
    allocation model by Jensen et al. (2022)

Version:
    3. Finished
    
Date:
    13-04-2023

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
### DATA IMPORT
# Daily stock returns
lDrop = ['excntry', 'ret']
          
dfDaily = pd.read_csv('usa_daily_return.csv', 
                      sep=',', 
                      index_col=False, 
                      low_memory=True,
                      usecols = lambda x: x not in lDrop)

# Observations from 1952-2020
dfDaily['date'] = pd.to_datetime(dfDaily['date'], format='%Y%m%d')
dfDaily['year'] = dfDaily['date'].dt.year
dfDaily = dfDaily[dfDaily['year'] <= 2020]
dfDaily = dfDaily[dfDaily['year'] >= 1942]

# Characteristics frame
dfChar = pd.read_csv('jensen2022_data.csv', 
                      sep=',', 
                      index_col=False, 
                      low_memory=True)

lKeep = ['id', 'eom', 'cowc_gr1a', 'oaccruals_at', 'oaccruals_ni', 
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

dfChar = dfChar[lKeep]

# Create the 13 cluster lists (Jensen et al Replication Crises pp 77-82)
# Accruals (excluding seas_16_20na)
lAccruals = ['cowc_gr1a', 'oaccruals_at', 'oaccruals_ni', 'taccruals_at', 
             'taccruals_ni']

# Debt Issuance (excluding capex_abn, debt_gr3, ni_ar1)
lDebtIss = ['fnl_gr1a', 'ncol_gr1a', 'nfna_gr1a', 'noa_at']

# Investment (excluding capx_gr2, capx_gr3, sale_gr3, saleq_gr1)
lInvest = ['aliq_at', 'at_gr1', 'be_gr1a', 'capx_gr1', 'coa_gr1a', 'col_gr1a',
           'emp_gr1', 'inv_gr1', 'inv_gr1a', 'lnoa_gr1a', 'mispricing_mgmt', 
           'ncoa_gr1a', 'nncoa_gr1a', 'noa_gr1a', 'ppeinv_gr1a', 'ret_60_12', 
           'sale_gr1', 'seas_2_5na']

# Low Leverage (excluding ni_ivol, rd_sale, rd5_at, z_score)
lLowLev = ['age', 'aliq_mat', 'at_be', 'bidaskhl_21d', 'cash_at', 'netdebt_me',
           'tangibility']

# Low Risk (excluding earnings_variability, ivol_hxz4_21d, ocfq_saleq_std,
# including rvol_252d) 
lLowRisk = ['beta_60m', 'beta_dimson_21d', 'betabab_1260d', 'betadown_252d',
            'ivol_capm_21d', 'ivol_capm_252d', 'ivol_ff3_21d', 'rmax1_21d', 
            'rmax5_21d', 'rvol_21d', 'rvol_252d', 'seas_6_10na', 
            'turnover_126d', 'zero_trades_21d', 'zero_trades_126d',
            'zero_trades_252d']

# Momentum (excluding resff3_6_1, resff3_12_1)
lMomentum = ['prc_highprc_252d', 'ret_3_1', 'ret_6_1', 'ret_9_1', 'ret_12_1', 
             'seas_1_1na']

# Profit Growth (excluding dsale_dinv, dsale_drec, dsale_dsga, niq_at_chg1,
# niq_be_chg1, niq_su, saleq_su)
lProfGrow = ['ocf_at_chg1', 'ret_12_7', 'sale_emp_gr1', 'seas_1_1an', 
             'tax_gr1a']

# Profitability (excluding f_score, niq_be)
lProfit = ['dolvol_var_126d', 'ebit_bev', 'ebit_sale', 'ni_be', 'o_score', 
           'ocf_at', 'ope_be', 'ope_bel1', 'turnover_var_126d']

# Quality (excluding dgp_dsale, ni_inc8q, niq_at, qmj, qmj_growth)
lQuality = ['at_turnover', 'cop_at', 'cop_atl1', 'gp_at', 'gp_atl1',
            'mispricing_perf', 'op_at', 'op_atl1', 'opex_at', 'qmj_prof', 
            'qmj_safety', 'sale_bev']

# Seasonality (excluding seas_16_20an, sti_gr1a)
lSeason = ['corr_1260d', 'coskew_21d', 'dbnetis_at', 'kz_index', 'lti_gr1a', 
           'pi_nix', 'seas_2_5an', 'seas_6_10an', 'seas_11_15an', 
           'seas_11_15na']

# Size (excluding rd_me)
lSize = ['ami_126d', 'dolvol_126d', 'market_equity', 'prc']

# Skewness/short-term reversal (excluding iskew_hxz4_21d)
lSkewness = ['iskew_capm_21d', 'iskew_ff3_21d', 'ret_1_0', 'rmax5_rvol_21d', 
             'rskew_21d']

# Value (excluding eqnetis_at, eqnpo_me, eqpo_me, ival_me, netis_at)
lValue = ['at_me', 'be_me', 'bev_mev', 'chcsho_12m', 'debt_me', 'div12m_me', 
          'ebitda_mev', 'eq_dur', 'eqnpo_12m', 'fcf_me', 'ni_me', 'ocf_me',
          'sale_me', 'intrinsic_value']

lClusters = [lAccruals, lDebtIss, lInvest, lLowLev, lLowRisk, lMomentum, 
             lProfGrow, lProfit, lQuality, lSeason, lSize, lSkewness, lValue]

lClusterNames = ['accruals', 'debt_iss', 'invest', 'low_lev', 'low_risk', 
                 'momentum', 'prof_grow', 'profit', 'quality', 'season', 
                 'size', 'skewness', 'value']

# Compute cross-sectional rank each month
for cluster in lClusters:
    for feature in cluster:
        dfChar[feature] = dfChar.groupby(['eom'])[feature].rank()
        
# Take average within-cluster rank
i = 0
for cluster in lClusters:
    sName = lClusterNames[i]
    dfChar[sName] = dfChar[cluster].mean(axis=1)
    i = i+1

# Standardize the clusters each month
for cluster in lClusterNames:
    dfChar[cluster] = dfChar.groupby('eom')[cluster].transform(
        lambda x: (x - x.mean()) / x.std())
    
dfChar = dfChar[['id', 'eom'] + lClusterNames]


# Split up datasets into blocks, since merging makes frame too big
dfChar['eom'] = pd.to_datetime(dfChar['eom'], format='%Y-%m-%d')
dfChar['mergemonth'] = dfChar['eom'].dt.month
dfChar['year'] = dfChar['eom'].dt.year
dfDaily['mergemonth'] = dfDaily['date'].dt.month

lYears = dfChar['year'].unique()

# Can merge on same month since characteristics were already 1 month lagged
os.chdir('C:\\Users\\eveli\\Documents\\ThesisData\\DailySplitYears')
for year in lYears:
    dfTemp1 = dfChar[dfChar['year'] == year]
    dfTemp2 = dfDaily[dfDaily['year'] == year]
    dfTemp2 = dfTemp2.merge(dfTemp1, how='left', on=['id', 'mergemonth'])
    dfTemp2.to_csv(f'daily_frame_{year}.csv', index=False)