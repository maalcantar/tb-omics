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
    fig, ax = plt.subplots(4, 1, figsize=(4, 4))
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

    
    
    
    