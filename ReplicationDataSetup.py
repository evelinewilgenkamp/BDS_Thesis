# -*- coding: utf-8 -*-
"""
ReplicationDataSetup.py

Purpose:
    Prepare all data necessary to replicate the paper by Jensen, Kelly,
    Malamud, & Pedersen (2022)

Version:
    1. Initialization
Date:
    13-03-2023

Author:
    Eveline Wilgenkamp
"""

###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import date, timedelta, datetime
from dateutil.parser import parse

###########################################################
### Working directory
os.chdir('C:\\Users\\eveli\\Documents\\ThesisData')

###########################################################
### IMPORTANT NOTE
# To prepare the dataset used by Jensen et al. (2022), we can use the 
# replicability code accessable at https://github.com/bkelly-lab/ReplicationCrisis
# SAS files extracted, ran in SAS Online through WRDS portal

###########################################################
### Data import
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
### Data cleaning
# Marketcap above 50th percentile of NYSE
dfChar = dfChar[(dfChar['size_grp'] == 'large') | (dfChar['size_grp'] == 'mega')]

# Select 115 stock characteristics from Jensen et al. 2022
dfChar['date'] = pd.to_datetime(dfChar['date'], format='%Y%m%d')
dfChar['year'] = dfChar['date'].dt.year
dfOld = dfChar[dfChar['year'] <= 1952].copy()




# Sample time: 1952-2020
dfChar = dfChar[dfChar['year'] <= 2020]
dfChar = dfChar[dfChar['year'] >= 1952]

###########################################################
### Data cleaning




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
