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

def compare_weights(weight_list, outfile, top_num=25):
    weights = [set(weight.iloc[:,0].values[:top_num]) for weight in weight_list]
    venn2(weights)
    plt.savefig(outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+', help='list of weight files to compare', 
                        required=True)
    parser.add_argument('-o', help='output file for plot', required=True)
    args = parser.parse_args()
    
    weight_files = args.i
    outfile = args.o

    weight_list = [pd.read_csv(weight_file) for weight_file in weight_files]   
    compare_weights(weight_list, outfile)
    
    
main()