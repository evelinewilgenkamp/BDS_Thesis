# -*- coding: utf-8 -*-
"""
CovarianceMatrix.py

Purpose:
    Estimate the (co)variance matrix needed for the portfolio allocation model
    by Jensen et al. (2022)

Version:
    1. Initialization
    
Date:
    11-04-2023

Author:
    Eveline Wilgenkamp
"""

# Data import


# Estimate latent factor returns
# r = S*f + e
# r: daily return
# S: matrix with 1 (intercept) and 13 cluster characteristics



# Computing S
# Each cluster is the average rank of the characteristics in the cluster
# Standardized as Z-score monthly



# Each trading day, estimate the cross-sectional regression



# Extract f hat



# Extract e hat



# Compute Var(fhat)
# Exponentially weighted (based on decays) sample covariance matrix
# Over past 10 years of daily observations



# Compute Var(ehat)
# Exponentially weighted (based on decays) moving acerage of squared residuals




