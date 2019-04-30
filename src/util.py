#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:44:21 2019

@author: calgunnarsson
"""

import pandas as pd
import numpy as np
import pickle


from sklearn.model_selection import train_test_split

def oneHotLabels(case_control):
    return [0 if label == 'control' else 1 for label in case_control]


def make_train_test(fl, datafile, test_size=0.20, random_state=42):
    
    with open(fl, 'rb') as f:
        features, labels = pickle.load(f)
        
    data = pd.read_csv(datafile)
    
    X_df = data.loc[:,features].dropna(axis=1)#
    y_df = data.loc[:, labels]
    y_df['group'] = oneHotLabels(y_df['group'])
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def make_test_group(te_inds, X_test, y_test):
    return X_test[te_inds], y_test[te_inds]

def summary_stats(ens):
    ens = ens.dropna(axis=0)
    cols_to_med = ['roc_auc', 'AP', 'accuracy'] 
    
    summary = pd.DataFrame()
    
    for col in cols_to_med:
        vals = ens[col].values
        med_idx = np.argsort(vals)[len(vals)//2]
        median = vals[med_idx]
        ci = np.percentile(vals, (2.5, 97.5))
        
        if np.isnan(median): #if nan
            continue
                
        summary[col+'_ci'] = [ci]
        summary[col+'_median'] = median
        
        
        med_row = ens.iloc[med_idx, :]
        if (col == 'roc_auc'):
            fpr = med_row['fpr']
            tpr = med_row['tpr']
            temp = pd.DataFrame({'fpr': [fpr], 'tpr': [tpr]})
            summary = pd.merge(summary, temp, how='left', 
                               left_index=True, right_index=True)
        if (col == 'AP'):
            precision = med_row['precision']
            recall = med_row['recall']
            temp = pd.DataFrame({'precision': [precision],
                                 'recall': [recall]})
            summary = pd.merge(summary, temp, how='left', 
                               left_index=True, right_index=True)
    return summary
        
def get_weights(weights_ens):
    weights_ens = weights_ens.abs()
    weights_ens = weights_ens.div(weights_ens.sum(axis=1), axis=0)
    weights_ens = weights_ens.sum(axis=0).transpose()
    weights_ens = pd.DataFrame(data=weights_ens, 
                               columns=['Feature importance'])
    #weights_ens = weights_to_pathways(weights_ens, chemfile)
    weights_ens = weights_ens.sort_values(by='Feature importance',
                                          ascending=False)
    return weights_ens
         
def to_array(str_array): 
    float_array = [float(string) for string in str_array[1:-1].split()]
    return float_array
