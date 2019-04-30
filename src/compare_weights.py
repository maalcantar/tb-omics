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

def weights_venn(weight_list, labels, outfile, top_num=25):
    weights = [set(weight.iloc[:,0].values[:top_num]) for weight in weight_list]
    
    venn2(weights, set_labels=labels)
    plt.savefig(outfile)

def weights_common(weight_list, labels, outfile, top_num=25):
    join = ['Biochemical','super.pathway', 'sub.pathway']
    #weight_list = [weight.reset_index() for weight in weight_list]
    weight_list = [weight.rename(columns={list(weight)[0]: model + ' Rank',
                   'Feature importance': model + ' Feature importance'}) 
                    for model, weight in zip(labels, weight_list)]
    weight_list = [weight[:top_num] for weight in weight_list]
    weights_common = pd.merge(weight_list[0], weight_list[1], 
                              on=['Biochemical','super.pathway', 'sub.pathway'])
    
    cols = join + ['Linear SVM Rank', 'Linear SVM Feature importance',
                   'RandomForest Rank', 'RandomForest Feature importance']
    weights_common = weights_common[cols]
    with open(outfile, 'w') as tf:
        tf.write(weights_common.to_latex())
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+', help='list of weight files to compare', 
                        required=True)
    parser.add_argument('-o', nargs=2, help='output files for plot and table', required=True)
    args = parser.parse_args()
    
    weight_files = args.i
    outfile = args.o
    labels = ['Linear SVM', 'RandomForest']
    weight_list = [pd.read_csv(weight_file) for weight_file in weight_files]   
    weights_venn(weight_list, labels, outfile[0])
    weights_common(weight_list, labels, outfile[1])
    
main()