#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:25:23 2019

@author: calgunnarsson
"""

import pickle
import argparse
import pandas as pd

from util import make_train_test, make_test_group
from util import summary_stats, get_weights

import sys 
import warnings

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)

def feat_SVM(X_train, y_train, model):
    model.fit(X_train, y_train)
    try: #for SVM 
        weights = pd.DataFrame([model.named_steps["clf"].coef_[0]], 
                                columns=X_train.columns[model.named_steps["fs"].get_support()])
    except AttributeError: #for RF
        try:
            weights = pd.DataFrame([model.feature_importances_],
                                   columns=X_train.columns)
        except AttributeError:
            weights = None 
    return weights

def pred_SVM(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    try: #for SVM
        y_score = model.predict(X_test)
        y_score = model.decision_function(X_test)
    except AttributeError: #for RF
        y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    #auc_ci = bootstrap_auc(model, X_train, X_test, y_train, y_test)
    
    averagePrecision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    
    modelAccuracy = model.score(X_test, y_test)
    results_df = pd.DataFrame({'fpr': [fpr], 'tpr': [tpr], 
                               'roc_auc': roc_auc,
                               'precision': [precision], 
                               'recall': [recall],
                               'AP': averagePrecision,
                               'accuracy': modelAccuracy})
    #print('The accuracy was:', modelAccuracy)
    return results_df

def pred_all(ntrials, fl, datafile, model, outfile, verbose=True):
    #make random train test splits to get ensemble simulations
    all_results_ens = []
    all_weights_ens = []
    for seed in range(ntrials):
        if (verbose and (seed % 10 == 0)):
            print('Running '+str(seed)+' of '+str(ntrials))
        X_train, X_test, y_train, y_test = make_train_test(fl, datafile, 
                                                           random_state=seed)
        label_train = y_train['group']
        label_test = y_test['group']
        
        all_results = pred_SVM(X_train, X_test, label_train, label_test, model)
        all_results['seed'] = seed
        all_results_ens.append(all_results)
        
        all_weights = feat_SVM(X_train, label_train, model)
        if all_weights is not None:
            all_weights['seed'] = seed
            all_weights_ens.append(all_weights)
    
    all_results_ens = pd.concat(all_results_ens, sort=False).set_index('seed')
    all_results_ens.to_csv(outfile+'_all_ens.csv')
    
    if all_weights_ens:
        all_weights_ens = pd.concat(all_weights_ens, sort=False).set_index('seed')
        #Normalize weights per row to get relative importances
        all_weights_ens = get_weights(all_weights_ens)
        all_weights_ens.to_csv(outfile+'_all_weights.csv')
     
    all_results_sum = summary_stats(all_results_ens)
    all_results_sum.to_csv(outfile+'_all_summary.csv')
    
    return all_results_sum

def pred_site(ntrials, fl, datafile, model, outfile, verbose=True):
    site_results_sum = []
    for site in ['AHRI', 'MAK', 'SUN', 'MRC']:
        #Accumulate simulation results for individual sites
        site_results_ens = []
        for seed in range(ntrials):
            if (verbose and (seed % 10 == 0)):
                print('Running '+str(seed)+' of '+str(ntrials))
            X_train, X_test, y_train, y_test = make_train_test(fl, datafile, 
                                                              random_state=seed)
            te_inds = y_test['site'].str.startswith(site)  
            X_te_s, y_te_s = make_test_group(te_inds, X_test, y_test)
            s_temp = pred_SVM(X_train, X_te_s, 
                             y_train['group'], y_te_s['group'], model)
            s_temp['seed'] = seed
            s_temp['site'] = site
            site_results_ens.append(s_temp)
        #Save simulation results to *_site_SITENAME.csv
        site_results_ens = pd.concat(site_results_ens, sort=False).set_index('seed') 
        site_results_ens.to_csv(outfile+'_site_ens'+site+'.csv')
        #Get median results
        site_sum_temp = summary_stats(site_results_ens)
        site_sum_temp['site'] = site
        #Accumulate 
        site_results_sum.append(site_sum_temp)          
     
    site_results_sum = pd.concat(site_results_sum, sort=False).set_index('site')
    site_results_sum.to_csv(outfile+'_site_summary.csv')

def pred_time(ntrials, fl, datafile, model, outfile, all_results_sum, verbose=True):
    #Split into <6 and >6 months to TB and run predictions separately
    time_results_sum = []
    time_all = all_results_sum
    time_all['time'] = 'all'
    time_results_sum.append(time_all)
    for time in ['proximal', 'distal']:
        time_results_ens = []
        for seed in range(ntrials):
            if (verbose and (seed % 10 == 0)):
                print('Running '+str(seed)+' of '+str(ntrials))
            #Make split and define subset to test
            X_train, X_test, y_train, y_test = make_train_test(fl, datafile, 
                                                               random_state=seed)
            te_inds = (y_test['time_to_tb'] < 6)
            if time == 'distal':
                te_inds = ~te_inds
            X_te_t, y_te_t = make_test_group(te_inds, X_test, y_test)
            
            t_temp = pred_SVM(X_train, X_te_t, 
                              y_train['group'], y_te_t['group'], model)
            t_temp['seed'] = seed
            t_temp['time'] = time
            time_results_ens.append(t_temp)
        #Save simulation results to *_time_TIMEPOINT.csv
        time_results_ens = pd.concat(time_results_ens, sort=False).set_index('seed') 
        time_results_ens.to_csv(outfile+'_time_ens'+time+'.csv')
        #Get median results
        time_sum_temp = summary_stats(time_results_ens)
        time_sum_temp['time'] = time
        #Accumulate 
        time_results_sum.append(time_sum_temp) 
        
    time_results_sum = pd.concat(time_results_sum, sort=False).set_index('time')
    time_results_sum.to_csv(outfile+'_time_summary.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='standardized measurements', 
                        required=True)
    parser.add_argument('-m', help='model file', required=True)
    parser.add_argument('-o', help='out file stem', required=True)
    parser.add_argument('-x', help='pickled features and labels',
                        default='fl.pkl')
    args = parser.parse_args()
    datafile = args.i
    modelfile = args.m
    outfile = args.o
    fl = args.x
    
    with open(modelfile, 'rb') as f:
        model = pickle.load(f)
    
    ntrials = 100
    
    all_results_sum = pred_all(ntrials, fl, datafile, model, outfile, verbose=True)
    #Split by site and run predictions separately 
    #Accumulate median results for individual sites
    
    _ = pred_site(ntrials, fl, datafile, model, outfile, verbose=True)

    _ = pred_time(ntrials, fl, datafile, model, outfile, 
                  all_results_sum, verbose=True)
    

     
main() 
    