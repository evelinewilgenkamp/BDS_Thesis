# -*- coding: utf-8 -*-
"""
ReplicationDataSetup.py

Purpose:
    Scrape all data necessary to replicate the paper by Jensen, Kelly,
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
from datetime import date, timedelta, datetime
from dateutil.parser import parse


###########################################################
### IMPORTANT NOTE
# To prepare the dataset used by Jensen et al. (2022), we can use the 
# replicability code accessable at https://github.com/bkelly-lab/ReplicationCrisis
# Their code is in R, we can translate this.


###########################################################
### Stock returns
# CRSP (waiting for WRDS reauthorization)


###########################################################
### Fundamentals
# Compustat (waiting for WRDS reauthorization)


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
