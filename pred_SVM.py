#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:25:23 2019

@author: calgunnarsson
"""

import pickle
import argparse
import numpy as np
import pandas as pd

from util import make_train_test, make_train_test_group
from util import plotROC, plotPRC

import sys 
import warnings

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)


def bootstrap_auc(model, X_train, X_test, y_train, y_test, nsamples=1000):
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        X_train_boot = X_train.values[idx, :]
        y_train_boot = y_train.values[idx]
        if len(np.unique(y_train_boot)) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        
        model.fit(X_train_boot, y_train_boot)
        try:
            pred = model.predict(X_test)
            pred = model.decision_function(X_test)
        except AttributeError:
            pred = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test.ravel(), pred.ravel())
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 97.5))

def pred_SVM(X_train, X_test, y_train, y_test, model):
    try: #for SVM
        y_score = model.predict(X_test)
        y_score = model.decision_function(X_test)
    except AttributeError: #for RF
        y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    auc_ci = bootstrap_auc(model, X_train, X_test, y_train, y_test)
    
    averagePrecision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    
    modelAccuracy = model.score(X_test, y_test)
    results_df = pd.DataFrame({'fpr': [fpr], 'tpr': [tpr], 
                               'roc_auc': roc_auc,
                               'auc_ci': [auc_ci],
                               'precision': [precision], 
                               'recall': [recall],
                               'AP': averagePrecision,
                               'accuracy': modelAccuracy})
    print('The accuracy was:', modelAccuracy)
    return results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='standardized measurements', 
                        required=True)
    parser.add_argument('-m', help='model file', required=True)
    parser.add_argument('-x', help='pickled features and labels',
                        default='fl.pkl')
    args = parser.parse_args()
    datafile = args.i
    modelfile = args.m
    fl = args.x
    
    with open(modelfile + '.pkl', 'rb') as f:
        model = pickle.load(f)
    
    X_train, X_test, y_train, y_test = make_train_test(fl, datafile)
    all_results = pred_SVM(X_train, X_test, y_train['group'], y_test['group'], model)

    all_results.to_csv(modelfile+'_all.csv')
    plotROC(all_results, modelfile+'_ROC_all.pdf', cond=False)
    plotPRC(all_results, modelfile+'_PRC_all.pdf', cond=False)
    
    ##Split by site and run predictions separately
    site_results = []
    for site in y_test['site'].unique():
        te_inds = y_test['site'].str.startswith(site)
        tr_inds = y_train['site'].str.startswith(site)
        X_tr_s, X_te_s, y_tr_s, y_te_s = make_train_test_group(tr_inds,
                                                               te_inds,
                                                               X_train,
                                                               X_test,
                                                               y_train,
                                                               y_test)
        s_temp = pred_SVM(X_tr_s, X_te_s, y_tr_s['group'], y_te_s['group'], model)
        s_temp['site'] = site
        site_results.append(s_temp)
    site_results = pd.concat(site_results, sort=False).set_index('site')
    site_results.to_csv(modelfile+'_site.csv')
    plotROC(site_results, modelfile+'_ROC_site.pdf')
    plotPRC(site_results, modelfile+'_PRC_site.pdf')
    
    ##Split into <6 and >6 months to TB and run predictions separately
    time_results = []
    time_all = all_results
    time_all['time'] = 'all'
    time_results.append(time_all)
    te_inds = (y_test['time_to_tb'] < 6)
    tr_inds = (y_train['time_to_tb'] < 6)
    for te_ind, tr_ind, time in zip([te_inds, ~te_inds],
                                    [tr_inds, ~tr_inds],
                                    ['proximal', 'distal']):
        X_tr_t, X_te_t, y_tr_t, y_te_t = make_train_test_group(tr_ind,
                                                               te_ind,
                                                               X_train,
                                                               X_test,
                                                               y_train,
                                                               y_test)
        t_temp = pred_SVM(X_tr_t, X_te_t, y_tr_t['group'], y_te_t['group'], model)
        t_temp['time'] = time
        time_results.append(t_temp)
    time_results = pd.concat(time_results, sort=False).set_index('time')
    time_results.to_csv(modelfile+'_time.csv')
    plotROC(time_results, modelfile+'_ROC_time.pdf')
    plotPRC(time_results, modelfile+'_PRC_time.pdf')
     
main() 
    