# Metabolomic signatures of tuberculosis progression

This repo contains the code to reproduce the analysis completed in "Metabolic profiling of tuberculosis-exposed patient cohort predicts progressor status." Our analyses use previously collected metabolome measurements from a cohort of tuberculosis-exposed individuals to (1) develop and optimize supervised machine learning approaches for predicting progression to active tuberculosis from metabolic profiles and (2) identify common metabolic signatures of tuberculosis progression. 

Collection of metabolomics data was described in Weiner et al.<a href="#note1" id="note1ref"><sup>1</sup></a> and can also be found on [Metabolomics Workbench](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Project&ProjectID=PR000666) but is included in `data/external` for completeness. 

## Reproducing analyses
The code is also accompanied by a Makefile, which can be used to re-run the analyses and re-make the figures and tables presented in the paper. 

Typing `make` or `make all` will pre-process data, run all analyses and create all figures, but individual analyses can be re-run:

* `pred_lin`: 
* `pred_rbf`:  

## Installation
First, install the required modules. To run all analyses in a conda virtual environment, create an empty `conda` environment, install `pip`, then use 
```bash
conda create -n tb-omics python=3.7
source activate tb-omics
conda install pip
pip install -r requirements.txt
```
This will create a virtual environment containing all the required modules needed to complete the analyses.

## Directory structure
### data
All data files are found in and/or will be written to `data/`
* `data/external`: all data externally downloaded, including `.csv` files containing metabolomics measurements, patient metadata and metabolite data
* `data/models`: results from training models, including hyperparameter optimization results, optimized models as `.pkl` files,
* `data/analysis`: data generated by running analysis scripts, including model evaluation results and feature importance scores for each model 

### source code
All code is in `src/`

* `load_data`: combines patient metabolite measurements with biochemical and patient metadata. Pre-processes resulting data, including dropping or imputing missing values and quantile standardization. Saves results to `data/analysis`.
* `train_SVM`: runs hyperparameter optimization for each model. Saves results of hyperparameter optimization and best estimator to `data/models`.
* `pred_SVM`: trains each model on an ensemble of random train-test splits and evaluates model performance based on the entire training set, as well as the training set subsetted by either region or time before onset of active tuberculosis. Saves raw and summarized results to `data/analysis`.
* `utils`: shared plotting and dataframe processing functions


## References


<a id="note1" href="#note1ref"><sup>1</sup></a> Weiner, J., Maertzdorf, J., Sutherland, J. S., Duffy, F. J., Thompson, E., Suliman, S., ... & Hanekom, W. A. (2018). *Metabolite changes in blood predict the onset of tuberculosis.* Nature communications, 9(1), 5208.

