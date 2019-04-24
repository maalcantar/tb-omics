#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:10:43 2019

@author: calgunnarsson
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import sys 
import warnings
import argparse

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)

def plotROC(fpr, tpr, roc_auc):
    lw = 2
    fig = plt.figure(dpi=400, figsize=(3,3))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM ROC')
    plt.legend(loc="lower right")
    plt.show()
    
def plotPRC(recall, precision, averagePrecision):
    fig = plt.figure(dpi=400, figsize=(3,3))
    step_kwargs = ({'step': 'post'}
                    if 'step' in signature(plt.fill_between).parameters
                    else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
              averagePrecision))
    plt.show()
    
def oneHotLabels(case_control):
    return [0 if label == 'control' else 1 for label in case_control]

# ways to improve:
# i) Hyperparmeter serach -- done ii) feature selection (serach for best threshold value) -- done iii) test models
# using different time points (currently just takes all casedand control data) iv) qualitative data?
# v) how much data do we need in train vs. test (+validation)
def SVM_model(data, features, labels, alpha=0.001, plot=True, kernel='linear'):
    # assigning labels to control (0) and case (1) subjects
    case_control = oneHotLabels(data['group']) 

    # data matrix will only consist of metabolite features (excluding other qualitative data)
    # dropping columns (features) with nan values -- most of these don't have enough info to impute
    X_df = data.loc[:,features].dropna(axis=1)#
    # labels matrix contains all metadata
    y_df = data.loc[:, labels]
    y_df['group'] = oneHotLabels(y_df['group'])
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.20, random_state=42)

    # running SVM
    threshold=0.01
    
    #params, CVscore, model = train_SVM(X_train, y_train, kernel=kernel, alpha=alpha, threshold=threshold)
    
    if kernel == 'linear':
        SVMFeat_df = pd.DataFrame([model.named_steps['clf'].coef_[0]], 
                     columns=fullSVM_df.columns[model.named_steps['fs'].get_support()])
    else:
        SVMFeat_df = []
        # calculating model accurarcy
    modelAccuracy = model.score(X_test, y_test)
    print('The accuracy was:', modelAccuracy)
    
    
    if plot:
        # ROC calculation
        y_score = model.predict(X_test)
        y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        averagePrecision = average_precision_score(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)

        # plotting ROC
        plotROC(fpr, tpr, roc_auc)

        # precision recall
        plotPRC(recall, precision, averagePrecision)

    return modelAccuracy, CVscore, SVMFeat_df


def train_SVM(X_train, y_train, 
              kernel='linear', alpha=0.001, threshold=0.01,
              save=False, output='SVM_optimized'):
    cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=4, random_state=42)
    if kernel == 'linear':
        ### ------- Everything in this block is for the hyperparameter sweep ------- ###
        # hinge loss for linear SVM; L1 for sparsity; don't shuffle data each epoch (reproducibility); 
        # balanced = don't bias classe (class imbalance) s;  best alpha found through parameter sweep -- manually 
        # setting in order to prevent long run time
        clf = linear_model.SGDClassifier(loss='hinge', penalty='l1', shuffle=False, class_weight='balanced' , alpha=alpha)
        featSelection = SelectFromModel(clf, threshold=threshold) # threshold for feature selection 
        # pipeline for SVM model -- this is still for cross-validation
        model = Pipeline([
                          ('fs', featSelection), 
                          ('clf', clf), 
                        ])
        # range of threshold values to test
        param_grid = {'fs__threshold': np.linspace(0.001, 25, num=25), 
                      'clf__alpha': np.logspace(-6, -2, num=6),
                      'clf__loss' : ['hinge', 'log', 'modified_huber']}
    else:
        # setting up non-linear model (either radial basis function or sigmoid)
        model = svm.SVC(kernel=kernel, C = 10.0, gamma=0.1)
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
             pickle.dump(grid, f)
         CVresults = pd.DataFrame(data=grid.cv_results_, 
                                  columns=['param_fs__threshold', 'param_clf__alpha',
                                           'param_clf__loss', 'mean_test_score'])
         CVresults.to_csv(output + '.csv')
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='standardized measurements', 
                        required=True)
    parser.add_argument('-o', help='output filename (hyperparam results + best model)', 
                        required=True)
    parser.add_argument('-x', help='pickled features and labels',
                        default='fl.pkl')
    parser.add_argument('-k', help='kernel type', default='linear')
    parser.add_argument('-mode', default='train')
    args = parser.parse_args()
    datafile = args.i
    fl = args.x
    modelfile = args.o
    kernel = args.k
    mode = args.mode
    
    with open(fl, 'rb') as f:
        features, labels = pickle.load(f)
    data = pd.read_csv(datafile)
    
    #case_control = oneHotLabels(data['group']) 

    # data matrix will only consist of metabolite features (excluding other qualitative data)
    # dropping columns (features) with nan values -- most of these don't have enough info to impute
    X_df = data.loc[:,features].dropna(axis=1)#
    y_df = data.loc[:, labels]
    y_df['group'] = oneHotLabels(y_df['group'])
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.20, random_state=42)
    
    
    if mode == 'train':
        # running SVM
        train_SVM(X_train, y_train['group'], kernel=kernel, save=True, output=modelfile) 
    else:
        with open(modelfile, 'rb') as f:
            grid = pickle.load(f)
        model = grid.best_estimator_
        
        #first group by region and predict
        #for ( ) in 
        
        #group by timepoint
        
    
main()
    
    