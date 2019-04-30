all: clean_data train pred figs

# File names for TB progressor vs control data
LTB_M = ./data/external/measurements_plasma_full.csv ./data/external/measurements_serum_full.csv ./data/external/measurements_plasmarpmi_full.csv
LTB_B = ./data/external/biochemicals_full_list_5.csv
LTB_P = ./data/external/full_unblinded_metadata_with_smoking_tst.csv
# Combined data file name
LTB_O = ./data/analysis/standardized_TB_metabolomes_LTB.csv
# Features and labels pickle
LTB_X = ./data/analysis/fl_ltb.pkl

clean_data: $(LTB_O) $(LTB_X)
	
$(LTB_O): $(LTB_M) $(LTB_B) $(LTB_P) ./src/load_data.py
	python ./src/load_data.py -m $(LTB_M) -b $(LTB_B) -p $(LTB_P) -o $@ -x $(LTB_X)

$(LTB_X): $(LTB_O)
	@if test -f $@; then :; else \
		rm -f $<; \
		make $<; \
	fi
	
# Grid search results pickle/csv
LTB_LIN = ./data/models/svm_ltb_lin.pkl
LTB_RBF = ./data/models/svm_ltb_rbf.pkl
LTB_RF = ./data/models/Random_Forest_Model.pkl
LTB_LIN_PARAM = ./data/models/svm_ltb_lin_hyperparam.csv
LTB_RBF_PARAM = ./data/models/svm_ltb_rbf_hyperparam.csv

train_lin: $(LTB_LIN) $(LTB_LIN_PARAM)
train_rbf: $(LTB_RBF) $(LTB_RBF_PARAM)
train: train_lin train_rbf

./data/models/svm_ltb_%.pkl: ./src/train_SVM.py $(LTB_O) $(LTB_X)
	python $< -i $(LTB_O) -o ./data/models/svm_ltb_$*_hyperparam.csv -m $@ -x $(LTB_X) -k $*

./data/models/svm_ltb_%_hyperparam.csv: ./data/models/svm_ltb_$%.pkl
	@if test -f $@; then :; else \
		rm -f $<; \
		make $<; \
	fi

svm_rbf_pred_all = ./data/analysis/svm_ltb_rbf_all_summary.csv
svm_rbf_pred_time = ./data/analysis/svm_ltb_rbf_time_summary.csv
svm_rbf_pred_site = ./data/analysis/svm_ltb_rbf_site_summary.csv
svm_lin_pred_all = ./data/analysis/svm_ltb_lin_all_summary.csv
svm_lin_pred_time = ./data/analysis/svm_ltb_lin_time_summary.csv
svm_lin_pred_site = ./data/analysis/svm_ltb_lin_site_summary.csv
rand_pred_all = ./data/analysis/Random_Forest_Model_all_summary.csv
rand_pred_time = ./data/analysis/Random_Forest_Model_time_summary.csv
rand_pred_site = ./data/analysis/Random_Forest_Model_site_summary.csv
# Weights results
WEIGHTS_RF = ./data/analysis/Random_Forest_Model_all_weights.csv
WEIGHTS_SVM = ./data/analysis/svm_ltb_lin_all_weights.csv

pred_rbf: $(svm_rbf_pred_all) $(svm_rbf_pred_site) $(svm_rbf_pred_time)
pred_lin: $(svm_lin_pred_all) $(svm_lin_pred_site) $(svm_lin_pred_time) $(WEIGHTS_SVM)
pred_rand: $(rand_pred_all) $(rand_pred_site) $(rand_pred_time) $(WEIGHTS_RF)
pred: $(pred_rbf) $(pred_lin) $(pred_rand)
./data/analysis/%_all_summary.csv: ./src/pred_SVM.py $(LTB_O) ./data/models/%.pkl $(LTB_X)
	python $< -i $(LTB_O) -m ./data/models/$*.pkl -x $(LTB_X) -o ./data/analysis/$*
%_site_summary.csv: %_all_summary.csv
	@if test -f $@; then :; else \
		rm -f $<; \
		make $<; \
	fi
%_time_summary.csv: %_site_summary.csv
	@if test -f $@; then :; else \
		rm -f $<; \
		make $<; \
	fi
%_weights.csv: %_summary.csv
	@if test -f $@; then :; else \
		rm -f $<; \
		make $<; \
	fi

svm_lin_roc_all = ./fig/figure2.roc.svm_ltb_lin_all.pdf
svm_lin_prc_all = ./fig/figure2.prc.svm_ltb_lin_all.pdf
svm_rbf_roc_all = ./fig/figure2.roc.svm_ltb_rbf_all.pdf
svm_rbf_prc_all = ./fig/figure2.prc.svm_ltb_rbf_all.pdf
rand_roc_all = ./fig/figure2.roc.Random_Forest_Model_all.pdf
rand_prc_all = ./fig/figure2.prc.Random_Forest_Model_all.pdf

fig2_lin: $(svm_lin_roc_all) $(svm_lin_prc_all)
fig2_rbf: $(svm_rbf_roc_all) $(svm_rbf_prc_all)
fig2_rand: $(rand_roc_all) $(svm_prc_all)
fig2: fig2_lin fig2_rbf fig2_rand
./fig/figure2.roc.%.pdf: ./src/figure.prc_roc.py ./data/analysis/%_summary.csv
	python $< -i ./data/analysis/$*_summary.csv -o $@ ./fig/figure2.prc.$*.pdf
./fig/figure2.prc.%.pdf: ./fig/figure2.roc.%.pdf
	@if test -f $@; then :; else \
		rm -f $<; \
		make $<; \
	fi

svm_lin_roc_site = ./fig/figure3.roc.svm_ltb_lin_site.pdf
svm_lin_prc_site = ./fig/figure3.prc.svm_ltb_lin_site.pdf
svm_lin_roc_time = ./fig/figure3.roc.svm_ltb_lin_time.pdf
svm_lin_prc_time = ./fig/figure3.prc.svm_ltb_lin_time.pdf
svm_rbf_roc_site = ./fig/figure3.roc.svm_ltb_rbf_site.pdf
svm_rbf_prc_site = ./fig/figure3.prc.svm_ltb_rbf_site.pdf
svm_rbf_roc_time = ./fig/figure3.roc.svm_ltb_rbf_time.pdf
svm_rbf_prc_time = ./fig/figure3.prc.svm_ltb_rbf_time.pdf
rand_roc_site = ./fig/figure3.roc.Random_Forest_Model_site.pdf
rand_prc_site = ./fig/figure3.prc.Random_Forest_Model_site.pdf
rand_roc_time = ./fig/figure3.roc.Random_Forest_Model_time.pdf
rand_prc_time = ./fig/figure3.prc.Random_Forest_Model_time.pdf

fig3_lin_site: $(svm_lin_roc_site) $(svm_lin_prc_site)
fig3_lin_time: $(svm_lin_roc_time) $(svm_lin_prc_time)
fig3_lin: fig3_lin_site fig3_lin_time
fig3_rbf_site: $(svm_rbf_roc_site) $(svm_rbf_prc_site)
fig3_rbf_time: $(svm_rbf_roc_time) $(svm_rbf_prc_site)
fig3_rbf: fig3_rbf_site fig3_rbf_time
fig3_rand_site: $(rand_roc_site) $(rand_prc_site)
fig3_rand_time: $(rand_roc_time) $(rand_prc_site)
fig3_rand: fig3_rand_site fig3_rand_time
fig3: fig3_lin fig3_rand fig3_rbf

./fig/figure3.roc.%_site.pdf: ./src/figure.prc_roc.py ./data/analysis/%_site_summary.csv
	python $< -i ./data/analysis/$*_site_summary.csv -o $@ ./fig/figure3.prc.$*_site.pdf --multi
./fig/figure3.roc.%_time.pdf: ./src/figure.prc_roc.py ./data/analysis/%_time_summary.csv
	python $< -i ./data/analysis/$*_time_summary.csv -o $@ ./fig/figure3.prc.$*_time.pdf --multi
./fig/figure3.prc.%.pdf: ./fig/figure3.roc.%.pdf
	@if test -f $@; then :; else \
		rm -f $<; \
		make $<; \
	fi

svm_lin_weights_top = ./fig/table2.svm_ltb_lin_top_weights.tex
rand_weights_top = ./fig/table2.Random_Forest_Model_top_weights.tex
table2_top: $(svm_lin_weights_top) $(rand_weights_top)

./fig/table2.%_top_weights.tex: ./src/table.weights.py ./data/analysis/%_all_weights.csv $(LTB_B)
	python $< -i ./data/analysis/$*_all_weights.csv -b $(LTB_B) -o $@


weights_venn = ./fig/figure4a.svm_rf_weights.pdf
weights_common = ./fig/table2.svm_rf_weights.tex
fig4a_venn: $(weights_venn) $(weights_common)
fig4: fig4a_venn
./fig/figure4a.%.pdf: ./src/compare_weights.py $(WEIGHTS_SVM) $(WEIGHTS_RF)
	python $< -i $(WEIGHTS_SVM) $(WEIGHTS_RF) -o $@ ./fig/table2.$*.tex
./fig/table2.%.tex: ./fig/figure4a.%.pdf
	@if test -f $@; then :; else \
		rm -f $<; \
		make $<; \
	fi

figs: fig2 fig3 fig4

