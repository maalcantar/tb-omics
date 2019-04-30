# Metabolic profiling of TB-exposed patient cohort predicts progressor status (20.440J Project)

## Project Overview
This repo contains all the code and data necessary to reproduce analyses from our 20.440 project entitled: "Metabolic profiling of TB-exposed patient cohort predicts progressor status." The goal of this study was to extract biomarker combinations using metabolomics data derived from TB-exposed individuals who participated in the Grand Challenges in Global Health GC6-74 project (Weiner et al. 2018). This data set consists of 751 case-control matched samples from TB-exposed patients with corresponding metadata and 1219 metabolite.

Various unsupervised and supervised machine learning methods are used in order to glean underlying metabolic patterns that can distinguish between TB-progressors and healthy individuals. We implement support vector machines (SVMs) with optimized hyperparamters and different kernel architectures (i.e., linear vs. radial basis function kernel) to predict progressor status from relative metabolite abundances. By using a sparse subset of metabolite features, we investigate how TB-progression surreptitiously manifests in subtle metabolite-level changes that cannot be captured by conventional biomarker detection methods.

This work was conducted by Cal Gunnarsson and Miguel Alcantar.

Graphical abstract:

![](Figures/TB_omics_graphical_abstract.png)

## Primary scripts and data files

The main data files used to conduct analyses can be found in the folder entitled <code>data</code>and include the following files:

* <code>measurements_plasma_full.csv </code>
    * Metabolomics measurements obtained from patient plasma -- this data is raw and not standardized
* <code>measurements_serum_full.csv </code>
    *   Metabolomics measurements obtained from patient serum -- this data is raw and not standardized
* <code>measurements_plasmarpmi_full.csv </code>
    *  Metabolomics measurements obtained from patient samples diluted in RPMI (this was done anytime enough sample for mass spectrometry could not be colleted -- this data is raw and not standardized
* <code> ParsedHMDB_v4.0.csv </code>
    *  Metabolites with their corresponding HMDB and pathway
* <code> biochemicals_full_list_5.csv</code>
    * Mapping between mass spectrometry identifier to actual biochemical name(e.g., M.11777 to glycine). Additional data includes HMDBs (manually curated for some), associated biochemical pathway, and mass-spec mode used to capture the metabolite (e.g., GC/MS)

Scripts are consolidated into various makefiles that can be run to automatically reprouce results. Primary makefiles include:
* <code>load_data.py </code>
    * Concatenates metabolomics data files (measurements_plasma_full, measurements_serum_full, measurements_plasmarpmi_full) and preprocesses data
    * Data preprocessing consists of quantile-standardization with respect to metabolites (Amaratunga and Cabrera 2011; Bolstad et al 2003). Metabolites were removed if greater than 10% of values were missing. Otherwise, data were imputed as the minimum
* <code> train_SVM.py </code>
   * Trains linear and RBF SVM using the full data set. 
      * this includes conducting a hyperparamter search to optimize the generalization capabilities of our model
   * The output of this script should be a trained linear and RBF SVM with optimized hyperparameters
* <code>pred_SVM.py</code>
   * Applies trained SVM to the full data set in order to classify TB-progressors vs. healthy controls
   * The output of this script should be ROC and PR curves with associated confidence intervals
   
Additionally, all figures created in the original manuscript can be found in the <code>Figues</code> folder.
    

## Repo structure

This repo consits of data files, found in the <code>data/external</code> folder, python scripts that are found in the home directory. Scripts automatically set the working directory to the data folder such that data can be easily accessed during analysis. 

## Reproducibility 

Results can be reproduced by running makefile in python from the terminal / command-line. For example <code> make Makefile </code> will automatically preprocess the data and produce all final figures

## Installation
