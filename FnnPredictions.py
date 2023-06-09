# -*- coding: utf-8 -*-
"""
FnnPredictions.py

Purpose:
    Predict returns of NYSE, AMEX, NASDAQ using FNNs
    Train: 1991-2000, valid: 2001-2002, test: 2003-2020, rolling window

Version:
    in progress
    
Date:
    28-06-2023

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
from sklearn.metrics import r2_score

# NN imports
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Tensorflow settings
np.random.seed(69)
tf.random.set_seed(69)
tf.config.set_visible_devices([], 'GPU') #dataset too big for gpu ram

# SHAP settings (to allow for batch normalization)
import shap
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough


###########################################################
### WORKING DIRECTORY
os.chdir('C:\\Users\\eveli\\Documents\\ThesisData')
#os.chdir('D:\\Eveline\\Downloads')


###########################################################
### DATA IMPORT (AND PREP)
### We must prep here, since an exported version is 15GB
### Takes less time to re-prep each time, than to import
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
def r2oos(y_test, y_pred_test):
    numerator = ((y_test - y_pred_test) ** 2).sum()
    denominator = (y_test ** 2).sum()
    r2 = 1 - (numerator / denominator)
    return r2


# model = modelBuilderNN3(pen, lr)
def modelBuilderNN3(pen, lr):
    # FNN so sequential model
    model = keras.Sequential()

    # Input layer with batch normalization
    model.add(layers.Input(shape=1109))
    model.add(layers.BatchNormalization())

    # Three ReLu layers with batch normalization
    model.add(layers.Dense(units=32, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=16, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=8, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    
    # Output layer
    model.add(layers.Dense(1, activation='linear'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse')

    return model

# model = modelBuilderNN4(pen, lr)
def modelBuilderNN4(pen, lr):
    # FNN so sequential model
    model = keras.Sequential()

    # Input layer with batch normalization
    model.add(layers.Input(shape=1109))
    model.add(layers.BatchNormalization())

    # Three ReLu layers with batch normalization
    model.add(layers.Dense(units=32, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=16, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=8, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=4, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    
    # Output layer
    model.add(layers.Dense(1, activation='linear'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse')

    return model

# model = modelBuilderNN5(pen, lr)
def modelBuilderNN5(pen, lr):
    # FNN so sequential model
    model = keras.Sequential()

    # Input layer with batch normalization
    model.add(layers.Input(shape=1109))
    model.add(layers.BatchNormalization())

    # Three ReLu layers with batch normalization
    model.add(layers.Dense(units=32, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=16, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=8, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=4, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=2, 
                           activation='relu',
                           kernel_regularizer=regularizers.l1(pen)))
    model.add(layers.BatchNormalization())
    
    # Output layer
    model.add(layers.Dense(1, activation='linear'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse')

    return model

# best_params = tuneNeuralNet(X_train, y_train, X_valid, y_valid)
def tuneNeuralNet(X_train, y_train, X_valid, y_valid, X_test, iLayer):
    # Tuning grid
    params_grid = {'lPenalty':[0.001, 0.0001, 0.00001],
                   'lLearnRate':[0.01, 0.001]}

    # Scores
    best_score = np.NINF
    best_params = None

    for lr in params_grid['lLearnRate']:
        for pen in params_grid['lPenalty']:
            # Initialize correct model
            if iLayer == 3:
                model = modelBuilderNN3(pen=pen, lr=lr)
            elif iLayer == 4:
                model = modelBuilderNN4(pen=pen, lr=lr)
            elif iLayer == 5:
                model = modelBuilderNN5(pen=pen, lr=lr)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, 
                                           restore_best_weights=True)

            # Fit model
            model.fit(X_train, y_train, epochs=100, batch_size=3000,
                      validation_data=(X_valid, y_valid),
                      callbacks=[early_stopping],
                      verbose=0)

            # Predictions on validation set
            y_pred_valid = model.predict(X_valid)[:, 0]

            # Validation set R^2
            score = r2_score(y_valid, y_pred_valid)
            
            # Backtesting print
            #print(f'Iteration with learning rate {lr} and l1 {pen} gave R^2 {score}')
            
            if score > best_score:
                # Set new best params
                best_score = score
                best_params = {'lr':lr, 'pen':pen}
                
                # Already do predictions for best params, saves retraining once
                y_pred_test = model.predict(X_test)[:, 0]
                
    # Return best parameters
    return best_params, y_pred_test

# y_pred = predictNeuralNet(X_train, y_train, X_valid, y_valid, X_test, y_test, 
#                           best_params, iLayer, importance=False)
def predictNeuralNet(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                     best_params, iLayer, importance=False):
    # 3 layer NN
    if iLayer == 3:
        model = modelBuilderNN3(pen=best_params['pen'], lr=best_params['lr'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=3000,
                  validation_data=(X_valid, y_valid),
                  callbacks=[early_stopping],
                  verbose=0)
        
    # 4 layer NN
    elif iLayer == 4:
        model = modelBuilderNN4(pen=best_params['pen'], lr=best_params['lr'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=3000,
                  validation_data=(X_valid, y_valid),
                  callbacks=[early_stopping],
                  verbose=0)
        
    # 5 layer NN
    elif iLayer == 5:
        model = modelBuilderNN5(pen=best_params['pen'], lr=best_params['lr'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=3000,
                  validation_data=(X_valid, y_valid),
                  callbacks=[early_stopping],
                  verbose=0)

    # Get predictions
    y_pred = model.predict(X_test)[:, 0]
    
    # Get SHAP values (for last model)
    if importance == True:
        bg = X_train.sample(100)
        deepexplainer = shap.DeepExplainer(model, bg)
        shaps = deepexplainer.shap_values(X_test.sample(1000).values)
    else:
        shaps = None
    
    return y_pred, shaps
    

###########################################################
### PREDICTIONS
# Set initial train, valid, test split
train_start = 1991
train_end = 2000
valid_start = 2001
valid_end = 2002
test_start = 2003
test_end = 2020

# List of neural net layers
lLayers = [5]

# List to store the predictions and best params in
lParamsNN3 = []
lParamsNN4 = []
lParamsNN5 = [] 

# List to store SHAP values
lShap = []

# Rolling window loop
for year in range(test_start, test_end+1):
    print(f'Training parameters: train start {train_start}, train end {train_end}, valid start {valid_start}, valid end {valid_end}, test set {year}')
    
    # Prepare train, validation, and test tests
    dfTest = dfData[dfData['year'] == year]
    
    X_train = dfData[(dfData['year'] >= train_start) & (dfData['year'] <= train_end)][lColumns]
    y_train = dfData[(dfData['year'] >= train_start) & (dfData['year'] <= train_end)]['ret_exc']
    X_valid = dfData[(dfData['year'] >= valid_start) & (dfData['year'] <= valid_end)][lColumns]
    y_valid = dfData[(dfData['year'] >= valid_start) & (dfData['year'] <= valid_end)]['ret_exc']
    X_test = dfTest[lColumns]
    y_test = dfTest['ret_exc']
    
    # Loop through the neural nets
    for layer in lLayers:
        print(f'NN{layer} tuning started for iteration: {year}')
        
        # Find the best parameters
        best_params, y_pred_test_1 = tuneNeuralNet(X_train, y_train, X_valid, 
                                                   y_valid, X_test, 
                                                   iLayer=layer)
        best_params['year'] = year
        
        if layer == 3:
            lParamsNN3.append(best_params)
        elif layer == 4:
            lParamsNN4.append(best_params)
        elif layer == 5:
            lParamsNN5.append(best_params)
    
        print(f'NN{layer} tuning finished for iteration: {year}')
        print(f'Best NN{layer} params for year {year} are {best_params}')
        print(f'NN{layer} predictions started for iteration: {year}')
        # 5 sets of predictions (ensemble) to average
        y_pred_test_2, shaps = predictNeuralNet(X_train, y_train, X_valid, 
                                                y_valid, X_test, y_test, 
                                                best_params, layer)
        
        y_pred_test_3, shaps = predictNeuralNet(X_train, y_train, X_valid, 
                                                y_valid, X_test, y_test, 
                                                best_params, layer)
        
        y_pred_test_4, shaps = predictNeuralNet(X_train, y_train, X_valid,
                                                y_valid, X_test, y_test,
                                                best_params, layer)
        
        y_pred_test_5, shaps = predictNeuralNet(X_train, y_train, X_valid, 
                                                y_valid,  X_test, y_test, 
                                                best_params, layer, 
                                                importance=True)
        
        # Compute average prediction (variance reduction)
        y_pred_avg = (y_pred_test_1 + y_pred_test_2 + y_pred_test_3 + y_pred_test_4 + y_pred_test_5) / 5
        
        # Get indices and dates to add to predictions
        dfPred = dfTest[['id', 'eom']].copy()
        dfPred['y_pred'] = y_pred_avg
        
        # Export the predictions
        dfPred.to_csv(f'NN{layer}_pred_{year}_3000batch.csv', index=False)
        
        # Test set R^2
        score = r2_score(y_test, y_pred_avg)
        print(f'For year {year} NN{layer} OOS R^2 is {score}')
        
        # Save SHAP values
        dfShap = pd.DataFrame(shaps[0], columns=lColumns)
        dfShap = dfShap.abs().mean()
        lShap.append(dfShap)
    
    # Set new rolling window
    train_start = train_start + 1
    train_end = train_end + 1
    valid_start = valid_start + 1
    valid_end = valid_end + 1


# R2 values
lFiles = []
for year in range(test_start, test_end+1):
    lFiles.append(f'NN3_pred_{year}_3000batch.csv')

dfPred = pd.concat(map(pd.read_csv, lFiles))
dfPred['eom'] = pd.to_datetime(dfPred['eom'], format='%Y-%m-%d')
dfPred = dfPred.merge(dfData[['id', 'eom', 'ret_exc']], on=['id','eom'])



