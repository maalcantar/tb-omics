#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:30:22 2019

@author: calgunnarsson
"""

import pandas as pd
import argparse

def weights_tables(weights, outfile, top_num=25):
    with open(outfile, 'w') as tf:
        tf.write(weights[:top_num].to_latex())

def weights_to_pathways(weights, chem, outfile, save=True):
    chem.columns = chem.columns.str.lower()
    chem = chem.rename(columns={'biochemical' : 'Biochemical'})
    chem = chem.set_index('Biochemical')
    chem = chem.loc[:, ['super.pathway', 'sub.pathway']]
    weights = weights.rename(columns={list(weights)[0] : 'Biochemical'})
    weights = weights.set_index('Biochemical')
    weights = pd.merge(weights, chem, on='Biochemical')
    weights = weights[~weights.index.duplicated(keep='first')]
    weights = weights.reset_index()
    weights.to_csv(outfile)
    return weights.sort_values(by='Feature importance', ascending=False)
     
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='weight file to convert to latex table', 
                        required=True)
    parser.add_argument('-b', help='biochemical information', required=True)
    parser.add_argument('-o', help='output file', required=True)
    args = parser.parse_args()
    
    weights = pd.read_csv(args.i)
    chem = pd.read_csv(args.b)
    weights = weights_to_pathways(weights, chem, args.i)
    print(weights)
    weights_tables(weights, args.o)
    
main()