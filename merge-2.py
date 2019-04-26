#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#OUTSTANDING ISSUES: THERE ARE DUPLICATE METABOLITES LOL 
import random
import numpy as np
import scipy as sp
import collections
import pandas as pd
from pylab import *
import seaborn as sns
import os
from matplotlib import pyplot as plt
from IPython.display import clear_output
import statsmodels.stats.multitest as multi
import sys
import warnings

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=UserWarning)

sns.set_style('white')
pd.options.display.float_format = '{:,.7f}'.format
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1500)

#set for reproducibility
#check current working directory

if (os.getcwd().split('\\')[-1] != 'data'):
    os.chdir('./data')
print(os.getcwd())


# In[ ]:





# In[ ]:


def load_metabolomics(filename):
    # loading in TB plasma metabolomics data from tab-delimted file to pandas dataframe
    df = pd.read_csv(filename)
    df = df.rename(columns={df.columns.values[0]: 'metabolite_name'})
    df = df.transpose()
    df.columns = df.iloc[0, :]
    df = df.iloc[1:, :]
    df.index.name = 'sample_id'
    
    return df


# In[ ]:


def impute(df, thresh=0.1):
    #retain columns with proportion missing values < threshold
    df = df[df.columns[df.isnull().mean() < thresh]]
    #impute remaining nans with minimum value
    df = df.apply(lambda x: x.fillna(x.min()), axis=0)
    return df.dropna(axis=1)


# In[ ]:


def load_patientmetadata(filename):
    # reading in patient metadata
    p_df = pd.read_csv(filename)
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


# In[ ]:


# 
def standardize_data(f_vals):
    from sklearn import preprocessing

    # applying standardization 
    scaler = preprocessing.QuantileTransformer()#StandardScaler()
    data_scaled = scaler.fit_transform(f_vals)
    
    return data_scaled


# In[ ]:


def make_df(f_vals, features, l_vals, labels):
    df = pd.concat([pd.DataFrame(data=l_vals, columns=labels), 
                    pd.DataFrame(data=f_vals, columns=features)], axis=1)
    return df


# In[ ]:


def perform_PCA(data, l_vals, labels, save=False, ncomp=10):
# computing principal components
    from sklearn import decomposition

    pcaAbs = decomposition.PCA(n_components=ncomp)
    data_PCA = pcaAbs.fit_transform(data)
    
    pc_cols = ['PC ' + str(i) for i in np.arange(1, ncomp + 1)]
    df_PCA = make_df(data_PCA, pc_cols, l_vals, labels)
    
    #Plot explained variance by number of components
    var_exp = pcaAbs.explained_variance_ratio_
    fig_ve, ax_ve = plt.subplots(1, 1)
    sns.lineplot(x=(np.arange(len(var_exp)) + 1), y=np.cumsum(var_exp), ax=ax_ve)
    plt.xlabel('PCA component number')
    plt.ylabel('Cumulative variance ratio')
    if save:
        plt.savefig('variance-exp.png', bbox_inches='tight', pad_inches=0.5)
    
    fig_pca, ax_pca = plt.subplots(1, 1)
    sns.scatterplot(x='PC 1', y='PC 2', data=df_PCA, hue='group', ax=ax_pca)
    
    return df_PCA


# In[ ]:


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
        stat, p_diff = sp.stats.ttest_ind(ctrl, case, nan_policy='omit', equal_var=(p_var < alpha_normal))
    else:
        stat, p_diff = sp.stats.ranksums(ctrl, case)
    
    return stat, p_diff


# In[ ]:


def significantMetabolites(ctrl, case, features, labels, alpha_normal=0.05, alpha_diff=0.05):
    pvals = []
    logfc = []
    stats = []
    for metab in features:
        metab_ctrl = ctrl[metab].values 
        metab_case = case[metab].values
        if (len(metab_ctrl.shape) > 1):
            display(metab)
        stat, p_diff = significanceTest(metab_ctrl, metab_case, alpha_normal=alpha_normal)
        pvals.append(p_diff)
        stats.append(stat)
        fc = np.mean(metab_case) / np.mean(metab_ctrl) #not matched in any way
        logfc.append(np.log2(fc))
    padj = multi.multipletests(pvals, alpha=alpha_diff, method='fdr_bh')
    
    significant = pd.DataFrame({'metabolite' : features, 'logFC' : logfc,
                                'statistic' : stats, 'P.Value' :  pvals, 'q' : padj[1]})
    
    return significant.sort_values(by='P.Value')


# In[ ]:


def convertHMDB(metabs, b_df):
    hmdb_dict = dict(zip(b_df['biochemical'], b_df['hmdb']))
    hmdb_ids  = [hmdb_dict[metab] for metab in metabs]
    return hmdb_ids


# In[ ]:


all_files = ['measurements_plasma_full.csv', 'measurements_serum_full.csv', 'measurements_plasmarpmi_full.csv']
all_df = []
for file in all_files:
    temp_df = impute(load_metabolomics(file))
    index = temp_df.index
    temp_df = pd.DataFrame(data=standardize_data(temp_df), columns=temp_df.columns)
    temp_df.index = index
    all_df.append(temp_df)
    
full_df   = pd.concat(all_df, sort=False).reset_index()
patient_df = load_patientmetadata('full_unblinded_metadata_with_smoking_tst.csv')
chem_df = load_biochemicaldata('biochemicals_full_list_5.csv')
full_df = combine_data(patient_df, full_df, chem_df)
full_df.to_csv('standardized_TB_metabolomes.csv')

labels = list(patient_df)
features = [x for x in list(full_df.columns) if x not in labels]
f_vals = full_df.loc[:, features].values
l_vals = full_df.loc[:, labels].values

# displaying shape and first few data entries
print('The shape of our data matrix is: ', full_df.shape)


# In[ ]:


full_df['metformin']#['hydroxy-CMPF*']


# In[ ]:





# In[ ]:


## HOW WELL DO SAMPLE PREPS CORRELATE?
#Extract donors for which there are multiple sample types at a given timepoint
dup_df = full_df[full_df.groupby(['donor_id', 'timepoint'])['sample_type'].transform('nunique') > 1] #ends up being only paired
#For each donor at each timepoint, calculate a correlation coefficient
dup_groups = dup_df.groupby(['donor_id', 'timepoint'])

corr = []
sig = []
donors = []
times = []
sample_types = []
for (donor, time), group in dup_groups:
    sample_types.append(group['sample_type'].values)
    donors.append(donor)
    times.append(time)
    
    shared_features = group[features[1:]].dropna(axis=1).T #drop columns that are not shared
    corr_temp, sig_temp = sp.stats.pearsonr(shared_features.values[:, 0], shared_features.values[:, 1])
    corr.append(corr_temp)
    sig.append(sig_temp)

corr_df = pd.DataFrame({'donor' : donors, 'timepoint' : times, 'sample_types' : sample_types, 
                        'Pearson correlation' : corr, 'p value' : sig, 'q value' : multi.multipletests(sig, method='fdr_bh')[1]})
display(corr_df)
#Result: all sample preps correlate significantly


# In[ ]:


#WHAT METABOLITES DIFFER SIGNIFICANTLY?

#Bin by time, to show when we can start detecting changes in the bulk population
full_df['time_bin'] = np.floor(np.abs(full_df['time_to_tb'] / 6)) #6 month increments
met_tp = []
for (timepoint), group in full_df.groupby(['time_bin']):
    #group = group
    ctrl = group[group['group'].str.contains('control')][features[1:]].dropna(axis=1)
    case = group[group['group'].str.contains('case')][features[1:]].dropna(axis=1)
    
    all_metabs = significantMetabolites(ctrl, case, list(ctrl), labels)
    sig_metabs = all_metabs[all_metabs['q'] <= 0.05]
    #print('Timepoint : ' + str(timepoint))
    #display(sig_metabs)
    met_tp.append(all_metabs)
    
#Result: We only see a large number of significant metabolites <6 months to TB
sig_metab = met_tp[0][met_tp[0]['q'] <= 0.05]
display(sig_metab)

#Plot these for all individuals, color by site


# In[ ]:


for i, df in enumerate(met_tp):
    sig = df[df['q'] <= 0.05]
    
    if len(sig) > 0:
        print('yes')
        with open('Diff_metab_table_'+str(i)+'.tex', 'w') as tf:
            tf.write(sig.to_latex())
     
print(len(met_tp[0]))
print(full_df['time_bin'].max())


# In[ ]:


#Violin plot of selected metabolites
#CMPFP 
#cortisol
#glutamine
#lactate
metabs_to_plot = ['3-CMPFP**', 'cortisol', 'glutamine', 'lactate']
proximate = full_df[full_df['time_bin'] == 0]

fig, ax = plt.subplots(4, 1, figsize=(4, 4))
for i, ax in enumerate(ax):
    
    sns.violinplot(y=proximate[metabs_to_plot[i]], x=proximate['group'], ax=ax)
    if (i < 4):
        ax.set_xlabel('')
        
fig.savefig('diff_metab_violin.pdf')


# In[ ]:





# In[ ]:


hmdb_ids = convertHMDB(sig_metab['metabolite'].values, chem_df)
sig_metab['hmdb'] = hmdb_ids
display(sig_metab)
len(sig_metab[sig_metab['logFC'] > 0])


# In[ ]:


near_tb = met_tp[0]
log2fc = near_tb['logFC']
log10p = -np.log10(near_tb['P.Value'])
sig = near_tb['q'] <= 0.05
fig, ax = plt.subplots(1, 1)
sns.scatterplot(x=log2fc.values, y=log10p.values, hue=sig.values, ax=ax)
ax.get_legend().remove()
ax.set_xlabel('Log$_{2}$ fold change')
ax.set_ylabel('Log$_{10}$ p value')
fig.savefig('Diff_metab_volcano.pdf')


# In[ ]:


#Convert diff exp metabs to tmod format
hmdb_ids = convertHMDB(near_tb['metabolite'].values, chem_df)
near_tb['hmdb'] = hmdb_ids

metab_ids = [chem_df.loc[i, 'id'] if pd.isnull(id) else id for i, id in enumerate(hmdb_ids)]
near_tb['metab_ids'] = metab_ids

near_tb = near_tb.drop_duplicates(subset='hmdb')
near_tb = near_tb.sort_values(by='P.Value')
near_tb = near_tb.reset_index(drop=True)
near_tb.to_csv('./tmod_diff.txt', sep='\t')

msea_metabs = near_tb.dropna(axis=0)
msea_metabs = msea_metabs.set_index('hmdb')
msea_metabs = msea_metabs.drop(columns=['metabolite', 'q', 'metab_ids'])
msea_metabs = msea_metabs.reset_index(drop=False)
msea_metabs = msea_metabs.rename({'hmdb':'feature'}, axis='columns')


msea_metabs.to_csv('./stats_table.txt', sep='\t')

msea_metabs.head()


# In[ ]:


met_tp_site = []
near_tb = full_df[full_df['time_bin'] == 0]
for (site), group in near_tb.groupby(['site']):
    ctrl = group[group['group'].str.contains('control')][features[1:]].dropna(axis=1)
    case = group[group['group'].str.contains('case')][features[1:]].dropna(axis=1)
    
    all_metabs = significantMetabolites(ctrl, case, list(ctrl), labels)
    sig_metabs = all_metabs[all_metabs['q'] <= 0.05]
    #print('Timepoint : ' + str(timepoint))
    display(sig_metabs)
    met_tp_site.append(all_metabs)


# In[ ]:


#WHAT METABOLITES CORRELATE WITH RISK? 
#analyzing separately by sample type (as broad as possible, color by location)
#spearman correlation with progressor status (y/n)
#pearson correlation with time to tb (metabolite-by-metabolite) 
#for a select few, show ones that go up, down, etc. relative to controls
all_case = full_df[full_df['group'].str.contains('case')].sort_values(by='case_group')
all_ctrl = full_df[full_df['group'].str.contains('control')].sort_values(by='case_group')


match_groups = np.unique([all_case['case_group'] + all_case['timepoint']]) #filter out 
all_groups = all_ctrl['case_group'] + all_ctrl['timepoint']
ctrl_keep = [(x in match_groups) for x in all_groups]

corr_bulk = []
sig_bulk = []
corr_matched = []
sig_matched = []
corr_biserial = []
sig_biserial = []
for metab in features[1:]:
    tb_time = all_case['time_to_tb']
    metab_values = all_case[metab] 
    
    metab_ratio = (all_case.groupby(['case_group', 'timepoint'])[metab].mean().values / 
                   all_ctrl[ctrl_keep].groupby(['case_group', 'timepoint'])[metab].mean().values)
    tb_time_matched = all_case.groupby(['case_group', 'timepoint'])['time_to_tb'].mean().values
    
    corr_bulk_temp, sig_bulk_temp = sp.stats.spearmanr(metab_values, tb_time, nan_policy='omit')
    corr_bulk.append(corr_bulk_temp)
    sig_bulk.append(sig_bulk_temp)
    
    corr_matched_temp, sig_matched_temp = sp.stats.spearmanr(metab_ratio, tb_time_matched, nan_policy='omit')
    corr_matched.append(corr_matched_temp)
    sig_matched.append(sig_matched_temp)
    
corr_df = pd.DataFrame({'metab' : features[1:], 
                        'Spearman_bulk' : corr_bulk, 'Spearman_matched' : corr_matched,
                        'p_bulk' : sig_bulk, 'p_matched' : sig_matched,
                        'q_bulk' : multi.multipletests(sig_bulk, method='fdr_bh')[1], 
                        'q_matched' : multi.multipletests(sig_matched, method='fdr_bh')[1]})
#display(corr_df_bulk.sort_values(by='q value'))

sig_corr_bulk = corr_df[corr_df['q_bulk'] <= 0.05]
sig_corr_matched = corr_df[corr_df['q_matched'] <= 0.05]
display(corr_df.sort_values(by='q_bulk'))

#In lumped analysis, there are some metabs that correlate with time to TB
#Venn diagram of significant metabolites that correlate with time to TB

#Progressor vs. control spearman R


# In[ ]:


display(sig_corr_matched)


# In[ ]:


hmdb_ids = convertHMDB(corr_df['metab'].values, chem_df)
corr_df['hmdb'] = hmdb_ids

metab_ids = [chem_df.loc[i, 'id'] if pd.isnull(id) else id for i, id in enumerate(hmdb_ids)]
corr_df['metab_ids'] = metab_ids

corr_df = corr_df.drop_duplicates(subset='hmdb')
corr_df = corr_df.sort_values(by='p_bulk')
corr_df = corr_df.reset_index(drop=True)
corr_df.to_csv('./tmod_corr.txt', sep='\t')


# In[ ]:


#WHAT'S THE SIGNAL TO NOISE RATIO?
#Within same individual, how does the metabolite change over time? (Means of std. dev, Pearson correlation)
#Identifying highly variable metabolites
#Pearson correlation between individuals in the same "case-control" match

all_ctrl = full_df[full_df['group'].str.contains('control')]

for (donor), group in all_ctrl.groupby('donor_id'):
    shared_features = group[features].dropna(axis=1).T #drop columns that are not shared
    
    if (shared_features.shape[1] > 1):
        corr_temp, sig_temp = sp.stats.pearsonr(shared_features.values[1:, 0], shared_features.values[1:, 1])
        display(corr_temp)


# In[ ]:


#CRASHES MY NOTEBOOK WHEN I GO BY DONOR LOL CRYING
#metab_var = all_ctrl.groupby('donor_id').apply(lambda x: np.std(x) / np.mean(x))
#display(metab_var)


# In[ ]:


weights = pd.read_csv('Weights.csv')
print(list(weights))
weights = weights.rename({list(weights)[0] : 'metab_ids'}, axis='columns')
hmdb_ids = convertHMDB(weights['metab_ids'].values, chem_df)
weights['hmdb'] = hmdb_ids

metab_ids = [chem_df.loc[i, 'id'] if pd.isnull(id) else id for i, id in enumerate(hmdb_ids)]
weights['metab_ids'] = metab_ids
weights.head()

weights.to_csv('./Weights_HMDB.csv')


# In[ ]:


def SVM_pred(data, features):
#Support vector machine
#classify  case vs. controls (vanilla version)
# ways to improve:
# i) Hyperparmeter serach ii) feature selection (serach for best threshold value) iii) test models
# using different time points (currently just takes all case and control data) iv) qualitative data?
# v) how much data do we need in train vs. test (+validation)

    # finding assigning labels to control (0) and case (1) subjects
    caseControlLabels = []
    for case_control in data['group']:
        if case_control == 'control':
            caseControlLabels.append(0)
        else:
            caseControlLabels.append(1)

    # extracting metabolite names -- these will be the features
    metabolites = []
    for feat in features:
        if 'M.' in feat:
            metabolites.append(feat)

    # data matrix will only consist of metabolite features (excluding other qualitative data)
    # dropping rows with nan
    fullSVM_df = data.loc[:,metabolites[0]:metabolites[-1]].dropna(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(fullSVM_df, caseControlLabels, test_size=0.33, random_state=42)

    # hinge loss for linear SVM; L1 for sparsity; don't shuffle data each epoch; don't bias classes; # fairly strong bias for L1
    clf = linear_model.SGDClassifier(loss="hinge", penalty="l1", shuffle=False, class_weight="balanced" , alpha=0.001)
    featSelection = SelectFromModel(clf, threshold=0.01) # threshold for feature selection 
    model = Pipeline([
                      ('fs', featSelection), 
                      ('clf', clf), 
                    ])
    model.fit(X_train, y_train)# train model
    
    # feature weightings
    SVMFeat_df = pd.DataFrame([model.named_steps["clf"].coef_[0]], 
                                    columns=fullSVM_df.columns[model.named_steps["fs"].get_support()])
    # calculating model accurarcy
    modelAccuracy = model.score(X_test, y_test)
    
    # ROC calculation
    y_score = model.predict(X_test)
    y_score = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # plotting ROC
    lw = 2
    plt.figure(dpi=400, figsize=(3,3))
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

    # precision recall
    plt.figure(dpi=400, figsize=(3,3))
    averagePrecision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
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


# In[ ]:


def SVM_pred_v2(data, features, alpha=0.001, threshold=0.01, plot = True, kernel = 'linear'):
#Support vector machine
#classify  case vs. controls (vanilla version)
# ways to improve:
# i) Hyperparmeter serach ii) feature selection (serach for best threshold value) iii) test models
# using different time points (currently just takes all casedand control data) iv) qualitative data?
# v) how much data do we need in train vs. test (+validation)

    # assigning labels to control (0) and case (1) subjects
    caseControlLabels = [0 if label == 'control' else 1 for label in data['group']]

    # extracting metabolite names --- these will be the features
    metabolites = features[1:]

    # data matrix will only consist of metabolite features (excluding other qualitative data)
    # dropping rows with nan
    fullSVM_df = data.loc[:,metabolites[0]:metabolites[-1]].dropna(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(fullSVM_df, caseControlLabels, test_size=0.33, random_state=42)

    if kernel == 'linear':
        # hinge loss for linear SVM; L1 for sparsity; don't shuffle data each epoch; don't bias classes; # fairly strong bias for L1
        clf = linear_model.SGDClassifier(loss='hinge', penalty='l1', shuffle=False, class_weight='balanced' , alpha=alpha)
        featSelection = SelectFromModel(clf, threshold=threshold) # threshold for feature selection 
        model = Pipeline([
                          ('fs', featSelection), 
                          ('clf', clf), 
                        ])
        model.fit(X_train, y_train)# train model

        # feature weightings
        SVMFeat_df = pd.DataFrame([model.named_steps['clf'].coef_[0]], 
                                  columns=fullSVM_df.columns[model.named_steps['fs'].get_support()])
        # calculating model accurarcy
        modelAccuracy = model.score(X_test, y_test)
        cv = min(caseControlLabels.count(0), caseControlLabels.count(0))
        scores = cross_val_score(clf, fullSVM_df, caseControlLabels, cv=cv)
        mean_cv = np.mean(scores) * 100
        print('The average CV score was:', mean_cv)
    elif kernel == 'rbf' or kernel == 'sigmoid':
        clf = svm.SVC(kernel=kernel, C = 10.0, gamma=0.1)
        param_grid = {'C': [0.0001, 0.001, 0.1, 1, 10, 100,1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10,100]}

        # Make grid search classifier
        model = GridSearchCV(svm.SVC(), param_grid, verbose=1)

        model.fit(X_train, y_train)# train model
        modelAccuracy = model.score(X_test, y_test) 
        cv = min(caseControlLabels.count(0), caseControlLabels.count(0))
        scores = cross_val_score(clf_grid, fullSVM_df, caseControlLabels, cv=3)
        mean_cv = np.mean(scores) * 100
        print('The average CV score was:', mean_cv)
        print('The model accuracy was:', modelAccuracy)

    if plot == True:
        # ROC calculation
        y_score = model.predict(X_test)
        y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)


        # plotting ROC
        lw = 2
        plt.figure(dpi=400, figsize=(3,3))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SVM ROC')
        plt.legend(loc='lower right')
        plt.show()

        # precision recall
        plt.figure(dpi=400, figsize=(3,3))
        averagePrecision = average_precision_score(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
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
    return modelAccuracy, mean_cv


# In[ ]:


def SVM_pred_v3(data, features, alpha=0.001, plot = True, kernel = 'linear'):
#Support vector machine
#classify  case vs. controls (vanilla version) -- done
# ways to improve:
# i) Hyperparmeter serach -- done ii) feature selection (serach for best threshold value) -- done iii) test models
# using different time points (currently just takes all casedand control data) iv) qualitative data?
# v) how much data do we need in train vs. test (+validation)

    # assigning labels to control (0) and case (1) subjects
    caseControlLabels = [0 if label == 'control' else 1 for label in data['group']]

    # extracting metabolite names --- these will be the features
    metabolites = features[1:]

    # data matrix will only consist of metabolite features (excluding other qualitative data)
    # dropping columns (features) with nan values -- most of these don't have enough info to impute
    fullSVM_df = data.loc[:,metabolites[0]:metabolites[-1]].dropna(axis=1)

    # splitting into training (80%) and testing sets (20%) -- cross validation will be used
    X_train, X_test, y_train, y_test = train_test_split(fullSVM_df, caseControlLabels, test_size=0.20, random_state=42)

    # running SVM
    threshold=0.01
    
    if kernel == 'linear':
        
        ### ------- Everything in this block is for the hyperparameter sweep ------- ###
        
        # cross validation for parameter sweeps
        cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3, random_state=42)
        
        # hinge loss for linear SVM; L1 for sparsity; don't shuffle data each epoch (reproducibility); 
        # balanced = don't bias classe (class imbalance) s;  best alpha found through parameter sweep -- manually 
        # setting in order to prevent long run time
        clf = linear_model.SGDClassifier(loss='hinge', penalty='l1', shuffle=False, class_weight='balanced' , alpha=alpha)
        
        # thresholding for feature selection
        featSelection = SelectFromModel(clf, threshold=threshold) # threshold for feature selection 
        
        # pipeline for SVM model -- this is still for cross-validation
        model = Pipeline([
                          ('fs', featSelection), 
                          ('clf', clf), 
                        ])
        # range of threshold values to test
        params = {'fs__threshold': np.linspace(0.001, 15, num=25)} # clf__alpha': np.logspace(-5, -3, num=3)

        # performing grid search
        grid = GridSearchCV(model, params, cv=cv)
        
        # fitting model
        grid.fit(fullSVM_df, caseControlLabels)
        CVscore = grid.best_score_
        print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, CVscore))
        
        ### ------- End hyperparameter sweep ------- ###
        
        ### ------- Running SVM model after hyperparameter sweep ------- ###
        # arguments same as above for the same reasons
        clf = linear_model.SGDClassifier(loss='hinge', penalty='l1', shuffle=False, class_weight='balanced' , alpha=alpha)
        featSelection = SelectFromModel(clf, threshold=grid.best_params_['fs__threshold']) # threshold for feature selection 
        model = Pipeline([
                          ('fs', featSelection), 
                          ('clf', clf), 
                        ])
        model.fit(X_train, y_train)# train model
#         sorted(pipeline.get_params().keys())

        SVMFeat_df = pd.DataFrame([model.named_steps['clf'].coef_[0]], 
                                  columns=fullSVM_df.columns[model.named_steps['fs'].get_support()])
        # calculating model accurarcy
        modelAccuracy = model.score(X_test, y_test)
        print('The accuracy was:', modelAccuracy)

    elif kernel == 'rbf' or kernel == 'sigmoid':
        
        # setting up non-linear model (either radial basis function or sigmoid)
        clf = svm.SVC(kernel=kernel, C = 10.0, gamma=0.1)
        
        # setting up grid for hyperparameter search
        param_grid = {'C': [0.0001, 0.001, 0.1, 1, 10, 100,1000], 'gamma': [0.00001, 0.001, 0.01, 0.1, 1, 10, 100]}

        # make grid search classifier (does hyperparameter sweep and cross-validates )
        model = GridSearchCV(clf, param_grid, verbose=1) # clf svm.SVC()

        # fitting and testing model
        model.fit(X_train, y_train)# train model
        modelAccuracy = model.score(X_test, y_test) 
#         cv = min(caseControlLabels.count(0), caseControlLabels.count(0))
#         scores = cross_val_score(model, fullSVM_df, caseControlLabels, cv=3)
#         mean_cv = np.mean(scores) * 100
        SVMFeat_df = []
        CVscore = model.best_score_
        print('The average CV score was:', CVscore)
        print('The model accuracy was:', modelAccuracy)

    if plot == True:
        # ROC calculation
        y_score = model.predict(X_test)
        y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)


        # plotting ROC
        lw = 2
        plt.figure(dpi=400, figsize=(3,3))
        plt.plot(fpr, tpr, color='darkorange',lw=lw) #, label='ROC curve (area = %0.2f)' % roc_auc
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SVM ROC')
        plt.legend(loc='lower right')
        plt.savefig('ROC' + kernel + '.pdf')
        plt.show()


        # precision recall
        plt.figure(dpi=400, figsize=(3,3))
        averagePrecision = average_precision_score(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
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
        plt.savefig('PRC' + kernel + '.pdf')

    return modelAccuracy, CVscore, SVMFeat_df


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm

fullData_SVM_dict = {'Linear Full Model': ['linear'], 
                     'RBF Full Model': ['rbf']}
# order will be [modelAccuracy, CVscore, SVMFeat_df]

for model_type, kernel_type in fullData_SVM_dict.items():
    accuracy, CVmeanScore, FeatureSelection  = SVM_pred_v3(full_df, features, alpha=0.001, 
                                                           plot = True, kernel = kernel_type[0])
    fullData_SVM_dict[model_type].append([accuracy, CVmeanScore, FeatureSelection])
    


# In[ ]:


fullData_SVM_dict = {'Linear Full Model': ['linear'], 
                     'RBF Full Model': ['rbf']}

for thing, key in fullData_SVM_dict.items():
    print(thing)
    print(key)


# In[ ]:


fullData_SVM_dict['Linear Full Model'][1][2].shape


# In[ ]:


# metabolites = features[1:]
sites = ['AHRI', 'MRC', 'SUN', 'MAK']

AHRI_df = full_df[full_df['site.gr'].str.startswith(sites[0])]
MRC_df = full_df[full_df['site.gr'].str.startswith(sites[1])]
SUN_df = full_df[full_df['site.gr'].str.startswith(sites[2])]
MAK_df = full_df[full_df['site.gr'].str.startswith(sites[3])]

acc = SVM_pred_v3(MAK_df, features, alpha=0.001, threshold=0.01, plot = True, kernel = 'rbf')

# # data matrix will only consist of metabolite features (excluding other qualitative data)
# # dropping rows with nan
# fullSVM_df = data.loc[:,metabolites[0]:metabolites[-1]].dropna(axis=1)


# In[ ]:


acc = SVM_pred_v3(MRC_df, features, alpha=0.001, threshold=0.01, plot = True, kernel = 'rbf')


# In[ ]:


acc = SVM_pred_v3(SUN_df, features, alpha=0.001, threshold=0.01, plot = True, kernel = 'rbf')


# In[ ]:


acc = SVM_pred_v3(MAK_df, features, alpha=0.001, threshold=0.01, plot = True, kernel = 'rbf')


# In[ ]:


set(full_df['time_to_tb'])


# In[ ]:


df = pd.read_csv('biochemicals_full_list_5.csv')
missingHMDB = []
for index, row in df.iterrows():
    if str(row['HMDB']) == 'nan' and row['BIOCHEMICAL'][0:3] != 'X -':
        missingHMDB.append(row['BIOCHEMICAL'])


# In[ ]:


df = pd.read_csv('biochemicals_full_list_5.csv')
missingHMDB = []
for index, row in df.iterrows():
    if str(row['HMDB']) == 'nan' and row['BIOCHEMICAL'][0:3] != 'X -':
        missingHMDB.append(row['BIOCHEMICAL'])


# In[ ]:


my_df = pd.DataFrame(missingHMDB)
my_df = my_df[~my_df[0].str.startswith('X -')]
my_df.to_csv('out.csv', index=False, header=False)


# In[ ]:




