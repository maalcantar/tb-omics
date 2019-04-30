#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:46:47 2019

@author: calgunnarsson
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
sns.set_style('white')

def weights_venn(weight_list, outfile, top_num=25):
    weights = [set(weight.iloc[:,0].values[:top_num]) for weight in weight_list]
    labels = ['Linear SVM', 'RandomForest']
    venn2(weights, set_labels=labels)
    plt.savefig(outfile)

def weights_common(weight_list, outfile, top_num=25):
    weights = [set(weight.iloc[:,0].values[:top_num]) for weight in weight_list]
    weights_list = [weight.reset_index(drop=False) for weight in weight_list]
    for i in range(len(weights) - 1):
        weights_intersect = weights[i] & weights[i + 1]
    print(weight_list[0].head())
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+', help='list of weight files to compare', 
                        required=True)
    parser.add_argument('-o', nargs=2, help='output files for plot and table', required=True)
    args = parser.parse_args()
    
    weight_files = args.i
    outfile = args.o

    weight_list = [pd.read_csv(weight_file) for weight_file in weight_files]   
    weights_venn(weight_list, outfile[0])
    weights_common(weight_list, outfile[1])
    
main()