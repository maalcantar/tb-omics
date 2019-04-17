
# File names for TB progressor vs control data
LTB_M = ./data/measurements_plasma_full.csv ./data/measurements_serum_full.csv ./data/measurements_plasmarpmi_full.csv
LTB_B = ./data/biochemicals_full_list_5.csv
LTB_P = ./data/full_unblinded_metadata_with_smoking_tst.csv
LTB_O = ./data/standardized_TB_metabolomes_LTB.csv
LTB_X = ./data/fl_ltb.pkl

load_ltb: $(LTB_M) $(LTB_B) $(LTB_P) load_data.py
	python load_data.py -m $(LTB_M) -b $(LTB_B) -p $(LTB_P) -o $(LTB_O) -x $(LTB_X)
