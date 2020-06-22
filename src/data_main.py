"""
Master script for running the indivdual parts of the data preparation and data preprocessing.
"""

import os
import data_preprocessing
import converter_fulldata_to_scikitmll_sparse

############################ setup preliminaries: paths, variables, etc. ###############################################

rootdir = "insert your project root path here"
rawdata_dir = rootdir + "raw_data/"  # where the raw data is sitting

targetdir = rootdir + "data/"  # for the cleaned datasets

sourcedir_codes = rootdir + "codes/"  # for the intermediate codes
sourcedir_verbatims = rootdir + "verbatims/"  # for the intermediate verbatims
traintestdir = rootdir + "data/traintest/"  # for the cleaned, fixed train/test splits
dumped_predictions = rootdir + 'dumped_predictions/'  # for storing model predictions

plot_folder = rootdir + "plots/"

folders = [rawdata_dir, targetdir, targetdir_csmith, sourcedir_codes, sourcedir_verbatims, traintestdir,
           plot_folder, plot_folder_csmith, metrics_folder, model_folder]

for folder in folders:
    if not os.path.exists(os.path.join(rootdir, folder)):
        os.mkdir(os.path.join(rootdir, folder))

############################# call the scripts / execute the analysis ##################################################
# list all the scripts in the right order here when done writing all the code

# 1. first the data preprocessing script, which calls anes_data_preparation
print("Starting with data preparation and the final data preprocessing for clean datasets!")
checkmate_preproc_main = data_preprocessing.main_preproc(rawdata_dir=rawdata_dir, targetdir=targetdir,
                                                         traintestdir=traintestdir, sourcedir_codes=sourcedir_codes,
                                                         sourcedir_verbatims=sourcedir_verbatims,
                                                         plot_folder=plot_folder)
if checkmate_preproc_main:
    print("Sucess on the preprocessing!")

# Has to be disabled after running once, as the splits cannot easily repeated!!!
# This has been finalized on: 05.04.2020 all data/all the splits are from this date

######################### OPTIONAL BLOCK: Additional data processing to sparse format ##################################
"""Please note that this module (below) consumes the complete datasets 1-10 (not split into train/test) such that the
interested user can convert the datasets of interest and then process them (including cross validation) how he/she sees
fit."""

checkmate_sparse = converter_fulldata_to_scikitmll_sparse.main_converter(data_source_dir_normal=targetdir,
                                                                         output_path=sparse_data_dir)

if checkmate_sparse:
    print("Done with the optional block to converting the data to scikit-multi format!")
