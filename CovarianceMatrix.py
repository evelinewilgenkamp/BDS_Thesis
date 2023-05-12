# -*- coding: utf-8 -*-
"""
CovarianceMatrix.py

Purpose:
    Estimate the (co)variance matrix needed for the portfolio allocation model
    by Jensen et al. (2022)

Version:
    2. Work in process
    
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
import statsmodels.formula.api as smf
import statsmodels.api as sm
import time
import os
from datetime import date, timedelta, datetime
from dateutil.parser import parse
from sklearn.preprocessing import minmax_scale

###########################################################
### WORKING DIRECTORY
os.chdir('C:\\Users\\eveli\\Documents\\ThesisData\\DailySplitYears')

###########################################################
### PREPARE FRAMES FOR RESULTS
# Covariates list
lClusters = ['constant', 'accruals', 'debt_iss', 'invest', 'low_lev', 
             'low_risk', 'momentum', 'prof_grow', 'profit', 'quality', 
             'season', 'size', 'skewness', 'value']

# Latent factor returns frame
dfFacRet = pd.DataFrame(columns=(['date'] + lClusters))

# Residuals frame
dfResid = pd.DataFrame()


###########################################################
### FUNCTIONS
# Function that cleans a yearly dataframe to prep for daily regressions
def cleanYearlyFrame(dfYear):
    """
    Purpose:
        Clean the yearly return-cluster frame
    Inputs:
        dfYear: dataframe with return-cluster data for 1 year
    Return value:
        dfYearClean: cleaned dataframe
        lDates: list with dates contained in the yearly frame, strings
    """
    # Copy to clean
    dfYearClean = dfYear.copy()
    
    # Keep only relevant columns    
    lKeep = ['id', 'date', 'ret_exc', 'accruals', 'debt_iss', 'invest', 
             'low_lev', 'low_risk', 'momentum', 'prof_grow', 'profit', 
             'quality', 'season', 'size', 'skewness', 'value']
    dfYearClean = dfYearClean[lKeep]
    
    # Remove missing values (either returns or one of the cluster signals)
    # Drops alot since we only use stocks above the 50th percentile NYSE Marketcap
    dfYearClean = dfYearClean[~dfYearClean.isnull().any(axis=1)]
    
    # Add a constant row to the frame
    dfYearClean['constant'] = 1

    # Make day variable string for easy parsing
    dfYearClean['date'] = dfYearClean['date'].astype(str)
    
    # Set ID as index (for future Series parsing)
    dfYearClean.set_index('id', inplace=True)
    
    # Create list with all dates in the frame to loop over
    lDates = dfYearClean['date'].unique()
    
    return dfYearClean, lDates



# Function that runs regression and stores relevant outcomes
def runDailyRegression(sDay, dfDay, lClusters, dfFacRet, dfResid):
    """
    Purpose:
        Run a cross-sectional regression of stock returns on cluster
        characteristics, store latent factor returns and error terms
    Inputs:
        day: string indicating the day
        dfDay: return dataframe for 1 day
        lClusters: list with covariates
    Return value:
        dfFacRet: dataframe with latent factor returns for days inserted
    """
    
    ### RUN REGRESSION
    # Run daily regression
    res = sm.OLS(endog=dfDay['ret_exc'], exog=dfDay[lClusters]).fit()

    ### STORE THE RESULTS
    # Latent factor returns
    lFacRet = res.params.append(pd.Series([sDay], index=['date']))
    dfFacRet = dfFacRet.append(lFacRet, ignore_index=True)
    
    # Residuals
    tempResid = res.resid.to_frame().T
    tempResid.insert(0, 'date', sDay)
    dfResid = dfResid.append(tempResid, ignore_index=True)
    
    return dfFacRet, dfResid
    


###########################################################
### RUNNING REGRESSIONS
# List of all relevant yearly frames to loop through
lYearFrames = []
for year in range(1950, 2021):
    lYearFrames.append(f'daily_frame_{year}.csv')

# Loop through years
for yearFrame in lYearFrames:
    # Load in yearly frame
    dfYear = pd.read_csv(yearFrame, index_col=False)
    
    # Clean the frame
    dfYearClean, lDates = cleanYearlyFrame(dfYear=dfYear)
    
    # Loop through days
    for sDay in lDates:
        dfDay = dfYearClean[dfYearClean['date'] == sDay]
        dfFacRet, dfResid = runDailyRegression(sDay=sDay, 
                                               dfDay=dfDay, 
                                               lClusters=lClusters, 
                                               dfFacRet=dfFacRet, 
                                               dfResid=dfResid)
        
# Export frames to prevent re-running
dfFacRet.to_csv('latent_facret.csv', index=False)
dfResid.to_csv('latent_resid.csv', index=False)        
    
###########################################################
### ESTIMATING (CO)VARIANCE MATRIX
### Compute Var_t(fhat_t+1)
# Exponentially-weightdd 
# pd.ewmstd(arg1=dffacret, halflife=)


# Compute Var(ehat)
# Exponentially weighted (based on decays) moving acerage of squared residuals




