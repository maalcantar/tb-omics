
# File names for TB progressor vs control data
LTB_M = ./data/external/measurements_plasma_full.csv ./data/external/measurements_serum_full.csv ./data/external/measurements_plasmarpmi_full.csv
LTB_B = ./data/external/biochemicals_full_list_5.csv
LTB_P = ./data/external/full_unblinded_metadata_with_smoking_tst.csv
# Combined data file name
LTB_O = ./data/analysis/standardized_TB_metabolomes_LTB.csv
# Features and labels pickle
LTB_X = ./data/analysis/fl_ltb.pkl
# Grid search results pickle/csv
LTB_LIN = ./data/analysis/svm_ltb_lin
LTB_RBF = ./data/analysis/svm_ltb_rbf
LTB_RF = ./data/analysis/Random_Forest_Model
# Accuracy results
LTB_LIN_ALL := $(LTB_LIN)_all.csv
LTB_LIN_SITE := $(LTB_LIN)_site.csv
LTB_LIN_TIME := $(LTB_LIN)_time.csv
LTB_RBF_ALL := $(LTB_RBF)_all.csv
LTB_RBF_SITE := $(LTB_RBF)_site.csv
LTB_RBF_TIME := $(LTB_RBF)_time.csv
LTB_RF_ALL := $(LTB_RF)_all.csv
LTB_RF_SITE := $(LTB_RF)_site.csv
LTB_RF_TIME := $(LTB_RF)_time.csv
# Weights results
WEIGHTS_RF := ./data/analysis/Random_Forest_Model_all_weights.csv
WEIGHTS_SVM := ./data/analysis/svm_ltb_lin_all_weights.csv
WEIGHTS_VENN := ./data/analysis/svm_rf_weights.pdf

.PHONY: all 
all: pred_lin pred_rbf pred_rand weights_venn

pred_lin: $(LTB_LIN_ALL)
pred_rbf: $(LTB_RBF_ALL)
pred_rand: $(LTB_RF_ALL)
weights_venn: $(WEIGHTS_VENN)
utils.py:

$(LTB_O): $(LTB_M) $(LTB_B) $(LTB_P) load_data.py
	python load_data.py -m $(LTB_M) -b $(LTB_B) -p $(LTB_P) -o $@ -x $(LTB_X)

$(LTB_X): $(LTB_O)
	@if test -f $@; then :; else \
		rm -f $<; \
		make $<; \
	fi

$(LTB_LIN).pkl: train_SVM.py $(LTB_O) $(LTB_X)
	python train_SVM.py -i $(LTB_O) -o $(LTB_LIN) -x $(LTB_X) -k linear
	
$(LTB_RBF).pkl: train_SVM.py $(LTB_O) $(LTB_X)
	python train_SVM.py -i $(LTB_O) -o $(LTB_RBF) -x $(LTB_X) -k rbf

$(LTB_RF).pkl:

#make hyperparam optimization reports for all models
%.csv: %.pkl
	@if test -f $@; then :; else \
		rm -f $<; \
		make $<; \
	fi
fl.csv: #ignore fl.pkl

%_all.csv: pred_SVM.py utils.py %.pkl
	python $< -i $(LTB_O) -x $(LTB_X)

$(LTB_LIN_ALL): pred_SVM.py utils.py $(LTB_LIN).pkl
	python pred_SVM.py -i $(LTB_O) -x $(LTB_X) -m $(LTB_LIN)

$(LTB_RBF_ALL): pred_SVM.py utils.py $(LTB_RBF).pkl
	python pred_SVM.py -i $(LTB_O) -x $(LTB_X) -m $(LTB_RBF)

$(LTB_RF_ALL): pred_SVM.py utils.py $(LTB_RF).pkl
	python pred_SVM.py -i $(LTB_O) -x $(LTB_X) -m $(LTB_RF)

$(WEIGHTS_VENN): compare_weights.py $(WEIGHTS_SVM) $(WEIGHTS_RF)
	python $< -i $(WEIGHTS_SVM) $(WEIGHTS_RF) -o $@
