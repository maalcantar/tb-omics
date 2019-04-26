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
from util import make_train_test
from util import plotROC, plotPRC
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

def bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test.ravel(), pred.ravel())
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 97.5))

def pred_SVM(X_test, y_test, model):
    try: #for SVM
        y_score = model.predict(X_test)
        y_score = model.decision_function(X_test)
    except AttributeError: #for RF
        y_score = model.predict_proba(X_test)[:, 1]
     
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    averagePrecision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    
    modelAccuracy = model.score(X_test, y_test)
    results_df = pd.DataFrame({'fpr': [fpr], 'tpr': [tpr], 
                               'roc_auc': roc_auc,
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
    all_results = pred_SVM(X_test, y_test['group'], model)
    all_results.to_csv(modelfile+'_all.csv')
    plotROC(all_results, modelfile+'_ROC_all.pdf', cond=False)
    plotPRC(all_results, modelfile+'_PRC_all.pdf', cond=False)
    ##Split by site and run predictions separately
    site_results = []
    for site in y_test['site'].unique():
        site_inds = y_test['site'].str.startswith(site)
        X_test_site = X_test[site_inds]
        y_test_site = y_test[site_inds]
        s_temp = pred_SVM(X_test_site, y_test_site['group'], model)
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
    prox_inds = (y_test['time_to_tb'] < 6)
    for time_inds, time in zip([prox_inds, ~prox_inds], ['proximal', 'distal']):
        X_test_time = X_test[time_inds]
        y_test_time = y_test[time_inds]
        t_temp = pred_SVM(X_test_time, y_test_time['group'], model)
        t_temp['time'] = time
        time_results.append(t_temp)
    time_results = pd.concat(time_results, sort=False).set_index('time')
    time_results.to_csv(modelfile+'_time.csv')
    plotROC(time_results, modelfile+'_ROC_time.pdf')
    plotPRC(time_results, modelfile+'_PRC_time.pdf')
     
main() 
    