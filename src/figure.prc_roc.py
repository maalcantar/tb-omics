#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:38:09 2019

@author: calgunnarsson
"""
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from util import to_array

def plotROC(df, filename, cond=True):
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(1, 1, 1)
    for ind, row in df.iterrows():
        fpr = to_array(row['fpr'])
        tpr = to_array(row['tpr'])
        if cond: 
            cond = ind + ', '
        else:
            cond = ''
        auc_ci = to_array(row['roc_auc_ci'])
       
        ax.plot(fpr, tpr,
                label=(cond+'AUC = %0.2f (95%%CI: %0.2f-%0.2f)' 
                      %(row['roc_auc_median'], auc_ci[0], auc_ci[1]))
                )
    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle=':')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    
def plotPRC(df, filename, cond=True):
    fig = plt.figure(dpi=400,)
    ax = fig.add_subplot(1, 1, 1)
    #step_kwargs = ({'step': 'post'}
    #                if 'step' in signature(plt.fill_between).parameters
    #                else {})
    #plt.step(recall, precision, color='b', alpha=0.2,
    #         where='post')

    for ind, row in df.iterrows():
        precision = to_array(row['precision'])
        recall = to_array(row['recall'])
        if cond:
            cond = ind + ', '
        else:
            cond = ''
        ap_ci = to_array(row['AP_ci'])
        ax.step(recall, precision, where='post',
                label=cond+'AP = %0.2f (95%%CI: %0.2f-%0.2f)'
                      %(row['AP_median'], ap_ci[0], ap_ci[1])
                )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    plt.title('PRC')
    plt.legend(loc='lower left')
    plt.savefig(filename)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='ensemble prc auc results', 
                        required=True)
    parser.add_argument('-o', nargs=2, help='model file', required=True)
    parser.add_argument('--multi', help='multiple curves', action='store_true')
    
    args = parser.parse_args()
    resultsfile = pd.read_csv(args.i)
    outfile = args.o
    cond = args.multi
    
    plotROC(resultsfile, outfile[0], cond=cond)
    plotPRC(resultsfile, outfile[1], cond=cond)

main()