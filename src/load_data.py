#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:06:16 2019

@author: calgunnarsson
"""

import pandas as pd
import argparse
import sys
import warnings
import pickle

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=UserWarning)

def load_metabolomics(filename):
    # loading in TB plasma metabolomics data from tab-delimted file to pandas dataframe
    df = pd.read_csv(filename)
    df = df.rename(columns={df.columns.values[0]: 'metabolite_name'})
    df = df.transpose()
    df.columns = df.iloc[0, :]
    df = df.iloc[1:, :]
    df.index.name = 'sample_id'
    
    return df

def impute(df, thresh=0.1):
    #drop columns with proportion missing values > threshold
    null_allowed = len(df.index) * thresh
    null_columns = df.columns.values[df.isnull().sum() > null_allowed]
    df = df.drop(columns=null_columns) 
    #retain columns with proportion missing values < threshold
    df = df[df.columns[df.isnull().mean() < thresh]]
    #impute remaining nans with minimum value
    df = df.apply(lambda x: x.fillna(x.min()), axis=0)
    return df.dropna(axis=1)

def load_patientmetadata(filename):
    # reading in patient metadata
    p_df = pd.read_csv(filename)
    p_df.columns = p_df.columns.str.lower()
    p_df.columns = p_df.columns.str.lower().str.rstrip()
    p_df = p_df.set_index('sample_id') 
    #drop redundant columns
    p_df = p_df.drop(columns=[p_df.columns.values[0], 'id'])
    
    return p_df

def load_biochemicaldata(filename):
    b_df = pd.read_csv(filename)
    b_df.columns = b_df.columns.str.lower()
    b_df = b_df.set_index('id')
    b_df = b_df.drop(columns=[b_df.columns.values[0]])
    return b_df.reset_index()

def combine_data(p_df, m_df, b_df):
    #join with full dataset
    m_df = m_df.set_index('sample_id').join(p_df)
    #rename columns with biologically meaningful metabolite names
    b_dict = dict(zip(b_df['id'], b_df['biochemical']))
    m_df = m_df.rename(b_dict, axis='columns')
    m_df = m_df.reset_index()
    #consolidate duplicated metabolites as maximum value across duplicates
    m_df_nodup = m_df.loc[:, ~m_df.columns.duplicated()]
    for dup in m_df.columns[m_df.columns.duplicated()].unique():
        temp = m_df[dup]
        m_df_nodup[dup] = temp.max(axis=1)
    #remove unindentified metabs    
    m_df_nodup = m_df_nodup.loc[:, ~m_df_nodup.columns.str.startswith('X -')]
    return m_df_nodup

def standardize_data(f_vals):
    from sklearn import preprocessing
    # applying standardization 
    scaler = preprocessing.QuantileTransformer()#StandardScaler()
    data_scaled = scaler.fit_transform(f_vals)
    return data_scaled

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', nargs='+', help='list of measurement files to combine', 
                        required=True)
    parser.add_argument('-b', help='biochemical information file', 
                        required=True)
    parser.add_argument('-p', help='patient metadata file', 
                        required=True)
    parser.add_argument('-o', help='output file to write data to', 
                        required=True)
    parser.add_argument('-x', help='file to pickle features and labels to',
                        default='fl.pkl')
    args = parser.parse_args()
    
    all_files = args.m  
    biochem_file = args.b
    patient_file = args.p
    out_file = args.o
    pickle_file = args.x
    
    #all_files = ['measurements_plasma_full.csv', 'measurements_serum_full.csv',
    #'measurements_plasmarpmi_full.csv']
    #biochem_file = 'biochemicals_full_list_5.csv'
    #patient_file = 'full_unblinded_metadata_with_smoking_tst.csv'
    #out_file = 'standardized_TB_metabolomes.csv'
    all_df = []
    for file in all_files:
        temp_df = impute(load_metabolomics(file))
        index = temp_df.index
        temp_df = pd.DataFrame(data=standardize_data(temp_df), columns=temp_df.columns)
        temp_df.index = index
        all_df.append(temp_df)
        
    full_df   = pd.concat(all_df, sort=False).reset_index()
    patient_df = load_patientmetadata(patient_file)
    chem_df = load_biochemicaldata(biochem_file)
    full_df = combine_data(patient_df, full_df, chem_df)
    full_df.to_csv(out_file)
    
    labels = list(patient_df)
    features = [x for x in list(full_df.columns) if x not in labels][1:]
    
    with open(pickle_file, 'wb') as f:
        pickle.dump([features, labels], f)
    
