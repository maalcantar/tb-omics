#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:01:15 2019

@author: calgunnarsson
"""
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.stats.multitest as multi
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

#calculate p-value on continuous data, selecting appropriate test based on normality & equal variance tests
def significanceTest(ctrl, case, alpha_normal=0.05):
    try:
        _, p_normal_ctrl = sp.stats.normaltest(ctrl, nan_policy='omit')
        _, p_normal_case = sp.stats.normaltest(case, nan_policy='omit')
    except:
        p_normal_ctrl = 1 
        p_normal_case = 1
        
    if (np.any(p_normal_ctrl < alpha_normal) and np.any(p_normal_case < alpha_normal)):
        _, p_var = sp.stats.bartlett(ctrl, case)
        _, p_diff = sp.stats.ttest_ind(ctrl, case, nan_policy='omit', equal_var=(p_var < alpha_normal))
    else:
        stat, p_diff = sp.stats.ranksums(ctrl, case)
    return stat, p_diff


def significantMetabolites(ctrl, case, features, alpha_normal=0.05, alpha_diff=0.05):
    pvals = []
    logfc = []
    stats = []
    for metab in features:
        metab_ctrl = ctrl[metab].values 
        metab_case = case[metab].values
        stat, p_diff = significanceTest(metab_ctrl, metab_case, alpha_normal=alpha_normal)
        pvals.append(p_diff)
        stats.append(stat)
        fc = np.mean(metab_case) / np.mean(metab_ctrl) 
        logfc.append(np.log2(fc))
    padj = multi.multipletests(pvals, alpha=alpha_diff, method='fdr_bh')
    significant = pd.DataFrame({'metabolite' : features, 'logFC' : logfc,
                                'statistic' : stats, 'P.Value' :  pvals, 'q' : padj[1]})
    
    return significant.sort_values(by='P.Value')

def plotMetabolites(metabs, df, filename, by):
    fig, ax = plt.subplots(len(metabs), 1, figsize=(4, 4))
    for i, ax in enumerate(ax):
        sns.violinplot(y=df[metabs[i]], x=df['group'], ax=ax)
        ax.set_xlabel('')
        
    fig.savefig(filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='standardized measurements', required=True)
    parser.add_argument('-x', help='pickled feature label file', default='fl.pkl')
    
    args = parser.parse_args()
    
    full_df = pd.read_csv(args.i)
    with open(args.x, 'rb') as f:
        features, labels = pickle.load(f)
        
    full_df['time_bin'] = np.floor(np.abs(full_df['time_to_tb'] / 6)) #6 month increments
    met_tp = []
    for (timepoint), group in full_df.groupby(['time_bin']):
        #group = group
        ctrl = group[group['group'].str.contains('control')][features[1:]].dropna(axis=1)
        case = group[group['group'].str.contains('case')][features[1:]].dropna(axis=1)
        
        all_metabs = significantMetabolites(ctrl, case, list(ctrl), labels)
        met_tp.append(all_metabs)
        
    #Plot these for all individuals, color by site
    
    for i, df in enumerate(met_tp):
        sig = df[df['q'] <= 0.05]
        
        if len(sig) > 0:
            print('yes')
            with open('Diff_metab_table_'+str(i)+'.tex', 'w') as tf:
                tf.write(sig.to_latex())
         
    print(len(met_tp[0]))
    print(full_df['time_bin'].max())
    
    #Violin plot of selected metabolites
    #CMPFP 
    #cortisol
    #glutamine
    #lactate
    metabs_to_plot = ['3-CMPFP**', 'cortisol', 'glutamine', 'lactate']
    proximate = full_df[full_df['time_bin'] == 0]
    
    fig, ax = plt.subplots(4, 1, figsize=(4, 4))
    for i, ax in enumerate(ax):
        
        sns.violinplot(y=proximate[metabs_to_plot[i]], x=proximate['group'], ax=ax)
        if (i < 4):
            ax.set_xlabel('')
            
    fig.savefig('diff_metab_violin.pdf')

main()
    
    
    
    