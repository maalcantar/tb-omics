#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:10:43 2019

@author: calgunnarsson
"""
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import sys 
import warnings
import argparse

from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from util import make_train_test

sns.set_style('white')

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)


def train_SVM(X_train, y_train, 
              kernel='linear', alpha=0.001, threshold=0.01,
              save=False, output='SVM_optimized'):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=4, random_state=42)
    if kernel == 'linear':
        # hinge loss for linear SVM; L1 for sparsity; don't shuffle data each epoch (reproducibility); 
        # balanced = don't bias classe (class imbalance) s
        clf = linear_model.SGDClassifier(loss='hinge', penalty='l1', shuffle=False, class_weight='balanced' , alpha=alpha)
        featSelection = SelectFromModel(clf, threshold=threshold) # threshold for feature selection 
        # pipeline for SVM model -- this is still for cross-validation
        model = Pipeline([
                          ('fs', featSelection), 
                          ('clf', clf), 
                        ])
        # range of threshold values to test
        param_grid = {'fs__threshold': np.linspace(0.001, 25, num=25), 
                      'clf__alpha': np.logspace(-6, -2, num=6)}
                      #'clf__loss' : ['hinge', 'log', 'modified_huber']}
    else:
        # setting up non-linear model (either radial basis function or sigmoid)
        model = svm.SVC(kernel=kernel, C = 10.0, gamma=0.1, cache_size=500)
        # setting up grid for hyperparameter search
        param_grid = {'C': np.logspace(-4, 3, num=7), 
                      'gamma': np.logspace(-5, 2, num=7)}
        # performing grid search
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_CVscore = grid.best_score_
    
    print("The best parameters are %s with a score of %0.2f"% (best_params, best_CVscore))
    
    pickle_file = output + '.pkl'
    if save:
        with open(pickle_file, 'wb') as f:
             pickle.dump(grid.best_estimator_, f)
             
        if kernel == 'linear':
            CVresults = pd.DataFrame(data=grid.cv_results_, 
                                     columns=['param_fs__threshold', 
                                              'param_clf__alpha', 
                                              'mean_test_score'])
        else:
            CVresults = pd.DataFrame(data=grid.cv_results_,
                                     columns=['param_C', 'param_gamma',
                                              'mean_test_score'])
    CVresults.to_csv(output + '.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='standardized measurements', 
                        required=True)
    parser.add_argument('-o', help='output filename', 
                        required=True)
    parser.add_argument('-x', help='pickled features and labels',
                        default='fl.pkl')
    parser.add_argument('-k', help='kernel type', default='linear')
    args = parser.parse_args()
    datafile = args.i
    fl = args.x
    outfile = args.o
    kernel = args.k
    
    X_train, X_test, y_train, y_test = make_train_test(fl, datafile)
    train_SVM(X_train, y_train['group'], kernel=kernel, save=True, output=outfile) 
    
main()
    
    