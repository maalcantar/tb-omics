#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:44:21 2019

@author: calgunnarsson
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

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

def plotROC(df, filename, cond=True):
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(1, 1, 1)
    for ind, row in df.iterrows():
        fpr = row['fpr']
        tpr = row['tpr']
        if cond: 
            cond = ind
        else:
            cond = ''
        ax.plot(fpr, tpr,
                label=cond+' (AUC = %0.2f)' % row['roc_auc'])
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
        precision = row['precision']
        recall = row['recall']
        if cond:
            cond = ind
        else:
            cond = ''
        ax.step(recall, precision, where='post',
                label=cond+' (AP = %0.2f)' %row['AP'])

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    plt.title('PRC')
    plt.legend(loc='lower right')
    plt.savefig(filename)