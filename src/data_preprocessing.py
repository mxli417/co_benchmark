"""
This is code to load the preparation functionality from the preparation script and
pull up the datasets in each of the categories already mentioned by Card/Smith. Then, a
first analysis of missings, code occurrences and possible problems with the
datasets are dealt with. Additionally, the datasets are preprocessed to only feature
used variables (no variable with only zeros later on!). Finally, the datasets are saved in a format
suitable for later processing with different machine-learning models, using fixed 90/10 train/test splits
and multi-label iterative stratification from Sechidis et al. (2011) implemented in
scikit-multilearn. Finalized on: 05.04.2020
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from anes_data_preparation import get_anes_data
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Verdana']


################################### SETUP HELPER FUNCTIONS #############################################################
def cleanup_data(input_df, killvars, kill_var=False, class_plot=False, saveplot=False, plot_output_folder='', fname=''):
    """Takes in a dataframe with prepared data (row code columns in binary format)
    generated by the anes_data_preparation script and cleans up rows with missing values
    on the verbatims (=> no verbatim recorded for that caseID), unused code columns (in debate, 21.01.2020),
    and specified variables (killvars) to produce a cleaned-up dataframe which then
    can be saved for further use. It optionally provides plots describing the class
    balance for each dataframe and can also save this code, provided a file name
    and setting saveplot to "True".

    :param input_df: dataframe prepared by the anes_data_preparation script
    :param killvars: variables which shall be dropped from dataframe
    :param kill_var: Switches on the deletion of the killvars
    :param class_plot: Switches on the plotting of class balances
    :param class_plot: Switches on the plotting of class balances
    :param saveplot: Switches on storing the plots on HDD
    :param plot_output_folder: Folder where the plots will be saved
    :param fname: Filname which will be used for the plot (with suffix: before/after depending on plot creation)
    :return: a clean dataset only containing coded verbatims without missings
    """
    # DEFENSE
    if input_df is None:
        raise NameError("Specified input dataframe does not exist! Please re-check!")

    if kill_var:
        if killvars is None:
            raise NameError("Specified variable list for deleting vars does not exist! Please re-check!")

    if saveplot:
        if not os.path.exists(plot_output_folder):
            raise NameError("Specified path for plot storage on HDD does not exist! Please re-check!")

    # MAIN FUNCTIONALITY
    # print out missings on verbatims for this dataframe
    print("Verbatim missings:", input_df['verbatim'].isna().sum())
    # subset to all observations with a non-NaN verbatim row entry
    input_df = input_df.dropna()
    input_df.reset_index(drop=True, inplace=True)  # refresh index in pandas
    # drop all columns with only zeros (unused codes)
    # TODO: document this in the thesis, Card/Smith did not do this! (See Email!)
    input_df = input_df.loc[:, input_df.any()]

    # iterate through dataset to weed out all obs with ONLY 1s in the 99/98 code category
    # as exclusive markers on 98 and 99 code-values indicate that the respondent either has
    # not answered or the answer is uninterpretable -> throw out
    selector_vars = list(input_df.columns)[2:]  # subset to the column code vars (first two are: caseID / verbatim)
    print("Code marker variables contained in dataframe: ", selector_vars)
    count_99, count_98, count_94, count_95, restricted_count_95, restricted_count_94 = 0, 0, 0, 0, 0, 0
    killrows = []
    zero_count = 0
    for index, row in input_df.iterrows():
        # get relevant content from row
        verbatim = row['verbatim']
        verbatim = verbatim.strip()
        row_content = row[selector_vars].values.tolist()
        row_content = np.asarray(row_content, dtype=int)
        if verbatim == "<DK>":
            count_94 += 1
            # print("DK found / row content: {}".format(row_content))

        if verbatim == "<RF>":
            count_95 += 1
            # print("RF found / row content: {}".format(row_content))

        # get nonzero positions from row
        nonzero_pos = np.nonzero(row_content)
        nonzero_sum = np.count_nonzero(row_content)
        # count all completely zero rows
        if nonzero_sum == 0:
            zero_count += 1
            killrows.append(index)  # append index of this line for later deletion

        # check if only 99 is marked in row -> mark for deletion
        if nonzero_sum == 1 and row_content[-1] == 1:  # 99
            count_99 += 1
            killrows.append(index)  # append index of this line for later deletion

        # check if only 98 is marked in row -> mark for deletion
        if nonzero_sum == 1 and row_content[-2] == 1:  # 98
            count_98 += 1
            killrows.append(index)  # append index of this line for later deletion

        # check if only 98 and 99 are marked in row -> mark for deletion
        if nonzero_sum == 2 and row_content[-1] == 1 and row_content[-2] == 1:
            count_99 += 1
            count_98 += 1
            killrows.append(index)  # append index of this line for later deletion

        # check restricted RF - via the combination of verbatim indicator, 95 indicator and 98 indicator
        if nonzero_sum == 2 and np.isin((row_content.shape[0] - 5), nonzero_pos):  # 95
            # and if verbatim is also <RF>
            if verbatim == "<RF>" and np.isin((row_content.shape[0] - 2), nonzero_pos):  # check additionally for 98:
                restricted_count_95 += 1  # this is an isolated 1 on the 95er codes

        # check unrestricted RF - via the combination of verbatim indicator, 95 indicator
        if nonzero_sum == 1 and np.isin((row_content.shape[0] - 5), nonzero_pos):  # 95
            # and if verbatim is also <RF>
            if verbatim == "<RF>":
                restricted_count_95 += 1  # this is an isolated 1 on the 95er codes

    # used for the final checks:
    print("Found: {} 99ers and: {} 98ers in dataframe and set to Nan".format(count_99, count_98))
    print("Found: {} rows are completely zero across all columns".format(zero_count))
    # throw out all observations with <RF> and 95 + 98 codes: 95 == REFUSE, 98 == NO UNIQUE CODE
    if len(killrows) > 0:
        print("Deleting the following rows: {}".format(killrows))
        input_df.drop(input_df.index[killrows], inplace=True)
        input_df.reset_index(drop=True, inplace=True)  # refresh index
    # Plot class balance before dataframe reduction
    if class_plot:
        # plot a the class balance before and after killing the special variables
        total_obs = input_df.shape[0]
        counter_df = input_df[selector_vars].apply(pd.Series.value_counts).fillna(0).astype(int)
        counter_df = counter_df.iloc[1, :]  # select only the occurrences / not the 0s
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(top=0.85)
        counter_df.plot(kind='bar', legend=False, ax=ax, alpha=0.8,
                        color='slateblue')  # also possible: colormap='Blues'
        # styling the plot
        fig.suptitle('Class balance before dataset reduction', fontsize=16)
        ax.set_title('D' + fname[1:] + ' (number of observations: ' + str(total_obs) + ')', fontsize=10, pad=15)
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        ax.set_ylabel('Number of observations')
        ax.set_xlabel('Label')
        ax.tick_params('x', labelrotation=90, pad=10)
        # remove top and right hand side box borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # restyle left and bottom axis
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        # print counts above the bars
        for p in ax.patches:
            an_text = ax.annotate(str(round(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', xytext=(0, 15), textcoords='offset points', rotation=0,
                                  arrowprops=dict(arrowstyle="-", connectionstyle="angle"))
            an_text.set_fontsize(6)
        # Store on HDD if requested
        if saveplot:
            filename = plot_output_folder + fname + '_before.png'
            fig.savefig(filename)
        plt.show()

    # Reduce dataframe by specified variables
    if kill_var:
        if isinstance(killvars, list):
            # kill special codes
            list_of_droppables = killvars
            print("List of variables to drop contains: ", list_of_droppables, ' with length: ', len(list_of_droppables))
            for l_element in list_of_droppables:
                # check if element exists in list
                if l_element in selector_vars:
                    selector_vars.remove(l_element)
            restored_selector_vars = ['caseID'] + ['verbatim'] + selector_vars
            # print("Selector vars restored: ", restored_selector_vars, " type: ", type(restored_selector_vars))
            input_df = input_df[restored_selector_vars]
    # Plot again after dataframe reduction
    if class_plot:
        total_obs = input_df.shape[0]
        counter_df = input_df[selector_vars].apply(pd.Series.value_counts).fillna(0).astype(int)
        counter_df = counter_df.iloc[1, :]
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(top=0.85)
        counter_df.plot(kind='bar', legend=False, ax=ax, alpha=0.8, color='slateblue')
        # styling the plot
        fig.suptitle('Class balance after dataset reduction', fontsize=16)
        ax.set_title('D' + fname[1:] + ' (number of observations: ' + str(total_obs) + ')', fontsize=10, pad=15)
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        ax.set_ylabel('Number of observations')
        ax.set_xlabel('Label')
        ax.tick_params('x', labelrotation=90, pad=10)
        # remove top and right hand side box borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # restyle left and bottom axis
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        # print counts above the bars
        for p in ax.patches:
            an_text = ax.annotate(str(round(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', xytext=(0, 15), textcoords='offset points', rotation=0,
                                  arrowprops=dict(arrowstyle="-", connectionstyle="angle"))
            an_text.set_fontsize(6)
        # Store on HDD if requested
        if saveplot:
            filename = plot_output_folder + fname + '_after.png'
            fig.savefig(filename)
        plt.show()
    # Final checks: NAs should be zero
    if (input_df.isna().sum().sum()) > 0:
        print(input_df)
        print("Missings still left: ", input_df.isna().sum())
        raise NameError('Missings could not be reduced to zero')
    # Print out stats for the remaining dataframe
    print("Processed dataframe shape: ", input_df.shape)
    return input_df


################################################ MAIN ANALYSIS #########################################################

# Please use this section if you cannot or do not want to use the master script
"""# 1. setup preliminaries: paths, variables, etc.
rawdata_dir = "C:/Users/mxmd/PycharmProjects/master_thesis/raw_data/"
targetdir = "C:/Users/mxmd/PycharmProjects/master_thesis/data/"  # for the cleaned datasets
traintestdir = "C:/Users/mxmd/PycharmProjects/master_thesis/data/traintest/"

sourcedir_codes = "C:/Users/mxmd/PycharmProjects/master_thesis/codes/"
sourcedir_verbatims = "C:/Users/mxmd/PycharmProjects/master_thesis/verbatims/"

plot_folder = "C:/Users/mxmd/PycharmProjects/master_thesis/plots/"""


def main_preproc(rawdata_dir, targetdir, traintestdir, sourcedir_codes, sourcedir_verbatims, plot_folder):
    """Takes in the main data folder paths and performs the data preprocessing which will create the cleaned
    datasets used in the final analysis.

    :param rawdata_dir: path to the rawdata
    :param targetdir: path to the folder where the clean data will be saved
    :param traintestdir: path to where the fixed train/test splits from the clean data will be saved
    :param sourcedir_codes: path to where the codes from the original data sit
    :param sourcedir_verbatims: path to where the verbatims from the original data sit
    :param plot_folder: path to where plots will be saved
    :return: Boolean, True if everything worked out
    """
    ########################### SECTION A: Preparing / Preprocessing the prepared data from ANES #######################

    # call the function from the previous script to load up and prepare the ANES 2008 datasets with verbatims and codes
    data_map = get_anes_data(rawdata_dir=rawdata_dir,
                             targetdir=targetdir,
                             targetdir_codes=sourcedir_codes,
                             targetdir_verbatims=sourcedir_verbatims,
                             optional_ds=False)

    # check contents of data map
    print("Number of datasets loaded: ", len(data_map))  # 10, as expected

    # create pool list for the map of cleaned datasets
    clean_data_map = []

    # we have to go through data with the code overview provided by ANES 2008, generate some visualizations for that,
    # generate new datasets without the missing values or uncodeable responses and count (for each Card/Smith dataset)
    # how many observations we have got, then calculate overall sum (-> then maybe contact Card/Smith regarding their
    # data loss during the preprocessing, how exactly their final datasets looked like and how many models they have
    # run)

    final_row_count = 0

    ########################## Dataset 1: checking missings, drop unsued codes, 99 98 etc. #############################

    # get dataset 1 and print
    preproc_dataset1 = data_map['Dataset_1']
    print("Dataset 1 df: \n", preproc_dataset1)

    # specify variables we want to drop here
    vars_to_drop = ['98', '99']  # really only drop variables which have no usable content

    # explanation: 98 is "no additional unique code", 99 "NA" hence we will drop both codes here, as these either
    # indicate no additional category or simply marks a verbatim which will be dropped nevertheless.

    # run cleanup function
    ds1 = cleanup_data(preproc_dataset1, vars_to_drop, True, True, True,
                       plot_output_folder=plot_folder, fname='dataset1')

    # inspect final dataset and save
    print("Cleaned dataset 1: ", ds1)
    print("Final number of codes used: ", (ds1.shape[1] - 2))
    ds1_fname = targetdir + "dataset1.csv"  # checked:
    ds1.to_csv(ds1_fname, index=False, header=True, encoding='utf-8')
    # append the data to the data map
    clean_data_map.append(ds1)
    print("Dataset 1: cleanup done!")

    # add the number of rows to the overall count
    final_row_count += ds1.shape[0]

    ########################## Dataset 2: checking missings, drop unsued codes, 99 98 etc. #############################

    preproc_ds2 = data_map['Dataset_2']
    print("Dataset 2 df: \n", preproc_ds2)

    # specify variables we want to drop here
    vars_to_drop = ['98', '99']  # really only drop variables which have no usable content

    # explanation: 98 is "no additional unique code", 99 "NA" hence we will drop both codes here, as these either
    # indicate no additional category or simply marks a verbatim which will be dropped nevertheless.

    # run cleanup function
    ds2 = cleanup_data(preproc_ds2, vars_to_drop, True, True, True,
                       plot_output_folder=plot_folder, fname='dataset2')

    # inspect final dataset and save
    print("Cleaned dataset 2: ", ds2)
    print("Final number of codes used: ", (ds2.shape[1] - 2))
    ds2_fname = targetdir + "dataset2.csv"  # checked:
    ds2.to_csv(ds2_fname, index=False, header=True, encoding='utf-8')
    # append the data to the data map
    clean_data_map.append(ds2)
    print("Dataset 2: cleanup done!")

    # add the number of rows to the overall count
    final_row_count += ds2.shape[0]

    ########################## Dataset 3: checking missings, drop unsued codes, 99 98 etc. #############################

    preproc_ds3 = data_map['Dataset_3']
    print("Dataset 3 df: \n", preproc_ds3)

    # specify variables we want to drop here
    vars_to_drop = ['98', '99']  # really only drop variables which have no usable content

    # explanation: 98 is "no additional unique code", 99 "NA" hence we will drop both codes here, as these either
    # indicate no additional category or simply marks a verbatim which will be dropped nevertheless.

    # run cleanup function
    ds3 = cleanup_data(preproc_ds3, vars_to_drop, True, True, True,
                       plot_output_folder=plot_folder, fname='dataset3')

    # inspect final dataset and save
    print("Cleaned dataset 3: ", ds3)
    print("Final number of codes used: ", (ds3.shape[1] - 2))
    ds3_fname = targetdir + "dataset3.csv"  # checked:
    ds3.to_csv(ds3_fname, index=False, header=True, encoding='utf-8')
    # append the data to the data map
    clean_data_map.append(ds3)
    print("Dataset 3: cleanup done!")

    # add the number of rows to the overall count
    final_row_count += ds3.shape[0]

    ########################## Dataset 4: checking missings, drop unsued codes, 99 98 etc. #############################

    preproc_ds4 = data_map['Dataset_4']
    print("Dataset 4 df: \n", preproc_ds4)

    # specify variables we want to drop here
    vars_to_drop = ['98', '99']  # really only drop variables which have no usable content

    # explanation: 98 is "no additional unique code", 99 "NA" hence we will drop both codes here, as these either
    # indicate no additional category or simply marks a verbatim which will be dropped nevertheless.

    # run cleanup function
    ds4 = cleanup_data(preproc_ds4, vars_to_drop, True, True, True,
                       plot_output_folder=plot_folder, fname='dataset4')

    # inspect final dataset and save
    print("Cleaned dataset 4: ", ds4)
    print("Final number of codes used: ", (ds4.shape[1] - 2))
    ds4_fname = targetdir + "dataset4.csv"  # checked:
    ds4.to_csv(ds4_fname, index=False, header=True, encoding='utf-8')
    # append the data to the data map
    clean_data_map.append(ds4)
    print("Dataset 4: cleanup done!")

    # add the number of rows to the overall count
    final_row_count += ds4.shape[0]

    ########################## Dataset 5: checking missings, drop unsued codes, 99 98 etc. #############################

    preproc_ds5 = data_map['Dataset_5']
    print("Dataset 5 df: \n", preproc_ds5)

    # specify variables we want to drop here
    vars_to_drop = ['98', '99']  # really only drop variables which have no usable content

    # explanation: 98 is "no additional unique code", 99 "NA" hence we will drop both codes here, as these either
    # indicate no additional category or simply marks a verbatim which will be dropped nevertheless.

    # run cleanup function
    ds5 = cleanup_data(preproc_ds5, vars_to_drop, True, True, True,
                       plot_output_folder=plot_folder, fname='dataset5')

    # inspect final dataset and save
    print("Cleaned dataset 5: ", ds5)
    print("Final number of codes used: ", (ds5.shape[1] - 2))
    ds5_fname = targetdir + "dataset5.csv"  # checked:
    ds5.to_csv(ds5_fname, index=False, header=True, encoding='utf-8')
    # append the data to the data map
    clean_data_map.append(ds5)
    print("Dataset 5: cleanup done!")

    # add the number of rows to the overall count
    final_row_count += ds5.shape[0]

    ########################## Dataset 6: checking missings, drop unsued codes, 99 98 etc. #############################

    preproc_ds6 = data_map['Dataset_6']
    print("Dataset 6 df: \n", preproc_ds6)

    # specify variables we want to drop here
    vars_to_drop = ['98', '99']  # really only drop variables which have no usable content

    # explanation: 98 is "no additional unique code", 99 "NA" hence we will drop both codes here, as these either
    # indicate no additional category or simply marks a verbatim which will be dropped nevertheless.

    # run cleanup function
    ds6 = cleanup_data(preproc_ds6, vars_to_drop, True, True, True,
                       plot_output_folder=plot_folder, fname='dataset6')

    # inspect final dataset and save
    print("Cleaned dataset 6: ", ds6)
    print("Final number of codes used: ", (ds6.shape[1] - 2))
    ds6_fname = targetdir + "dataset6.csv"  # checked:
    ds6.to_csv(ds6_fname, index=False, header=True, encoding='utf-8')
    # append the data to the data map
    clean_data_map.append(ds6)
    print("Dataset 6: cleanup done!")

    # add the number of rows to the overall count
    final_row_count += ds6.shape[0]

    ########################## Dataset 7: checking missings, drop unsued codes, 99 98 etc. #############################

    preproc_ds7 = data_map['Dataset_7']
    print("Dataset 7 df: \n", preproc_ds7)

    # specify variables we want to drop here
    vars_to_drop = ['98', '99']  # really only drop variables which have no usable content

    # explanation: 98 is "no additional unique code", 99 "NA" hence we will drop both codes here, as these either
    # indicate no additional category or simply marks a verbatim which will be dropped nevertheless.

    # run cleanup function
    ds7 = cleanup_data(preproc_ds7, vars_to_drop, True, True, True,
                       plot_output_folder=plot_folder, fname='dataset7')

    # inspect final dataset and save
    print("Cleaned dataset 7: ", ds7)
    print("Final number of codes used: ", (ds7.shape[1] - 2))
    ds7_fname = targetdir + "dataset7.csv"  # checked:
    ds7.to_csv(ds7_fname, index=False, header=True, encoding='utf-8')
    # append the data to the data map
    clean_data_map.append(ds7)
    print("Dataset 7: cleanup done!")

    # add the number of rows to the overall count
    final_row_count += ds7.shape[0]

    ########################## Dataset 8: checking missings, drop unsued codes, 99 98 etc. #############################

    preproc_ds8 = data_map['Dataset_8']
    print("Dataset 8 df: \n", preproc_ds8)

    # specify variables we want to drop here
    vars_to_drop = ['98', '99']  # really only drop variables which have no usable content

    # explanation: 98 is "no additional unique code", 99 "NA" hence we will drop both codes here, as these either
    # indicate no additional category or simply marks a verbatim which will be dropped nevertheless.

    # run cleanup function
    ds8 = cleanup_data(preproc_ds8, vars_to_drop, True, True, True,
                       plot_output_folder=plot_folder, fname='dataset8')

    # inspect final dataset and save
    print("Cleaned dataset 8: ", ds8)
    print("Final number of codes used: ", (ds8.shape[1] - 2))
    ds8_fname = targetdir + "dataset8.csv"  # checked:
    ds8.to_csv(ds8_fname, index=False, header=True, encoding='utf-8')
    # append the data to the data map
    clean_data_map.append(ds8)
    print("Dataset 8: cleanup done!")

    # add the number of rows to the overall count
    final_row_count += ds8.shape[0]

    ########################## Dataset 9: checking missings, drop unsued codes, 99 98 etc. #############################

    preproc_ds9 = data_map['Dataset_9']
    print("Dataset 9 df: \n", preproc_ds9)

    # specify variables we want to drop here
    vars_to_drop = ['98', '99']  # really only drop variables which have no usable content

    # explanation: 98 is "no additional unique code", 99 "NA" hence we will drop both codes here, as these either
    # indicate no additional category or simply marks a verbatim which will be dropped nevertheless.

    # run cleanup function
    ds9 = cleanup_data(preproc_ds9, vars_to_drop, True, True, True,
                       plot_output_folder=plot_folder, fname='dataset9')

    # inspect final dataset and save
    print("Cleaned dataset 9: ", ds9)
    print("Final number of codes used: ", (ds9.shape[1] - 2))
    ds9_fname = targetdir + "dataset9.csv"  # checked:
    ds9.to_csv(ds9_fname, index=False, header=True, encoding='utf-8')
    # append the data to the data map
    clean_data_map.append(ds9)
    print("Dataset 9: cleanup done!")

    # add the number of rows to the overall count
    final_row_count += ds9.shape[0]

    ########################## Dataset 10: checking missings, drop unsued codes, 99 98 etc. ############################

    preproc_ds10 = data_map['Dataset_10']
    print("Dataset 10 df: \n", preproc_ds10)

    # specify variables we want to drop here
    vars_to_drop = ['98', '99']  # really only drop variables which have no usable content

    # explanation: 98 is "no additional unique code", 99 "NA" hence we will drop both codes here, as these either
    # indicate no additional category or simply marks a verbatim which will be dropped nevertheless.

    # run cleanup function
    ds10 = cleanup_data(preproc_ds10, vars_to_drop, True, True, True,
                        plot_output_folder=plot_folder, fname='dataset10')

    # inspect final dataset and save
    print("Cleaned dataset 10: ", ds10)
    print("Final number of codes used: ", (ds10.shape[1] - 2))
    ds10_fname = targetdir + "dataset10.csv"  # checked:
    ds10.to_csv(ds10_fname, index=False, header=True, encoding='utf-8')
    # append the data to the data map
    clean_data_map.append(ds10)
    print("Dataset 10: cleanup done!")

    # add the number of rows to the overall count
    final_row_count += ds10.shape[0]

    ####################################################################################################################
    # count through the total number of verbatims which have been recorded with codes

    print("Overall observations count: ", final_row_count)  # last update: 28.466 observations on 05.04.2020

    # history: 28502 on 21.01.2020 / 28544 on 10.12.19 // original report Card&Smith: 28500
    ################################ Break out fixed train/test splits per dataset #####################################
    print("Creating the fixed train/test-splits from the preprocessed data with 10% of the data for the test dataset!")

    # check len of dataset map
    print("Len of dataset map:{}".format(len(clean_data_map)))

    # check if output folder for the datasets exists - create if not
    if not os.path.isdir(traintestdir):
        os.mkdir(traintestdir)

    # split every dataset into a train portion and test portion, keeping everything iteratively stratified, as
    # described by Sechidis et al. (2011), put 10% of the data into the test set, the rest into the train set
    # as described by Card/Smith (see page 4)

    dataset_counter = 1
    for dataset in clean_data_map:
        # print a status message
        print("Generating a fixed-size train/test split for datasets {}".format(dataset_counter))
        # extract the column names for the X and y matrices
        x_colnames = dataset.iloc[:, :2].columns.values.tolist()
        y_colnames = dataset.iloc[:, 2:].columns.values.tolist()
        # extract the feature matrix X and the label matrix y
        X_ds = dataset.iloc[:, :2].to_numpy()  # convert X to ndarray (important for the iterative stratification)
        y_ds = dataset.iloc[:, 2:].to_numpy(dtype=int)  # convert y label matrix to binary ndarray of integers
        # generate the split (still fails, maybe because of the X data type)
        # link to method: http://scikit.ml/api/skmultilearn.model_selection.iterative_stratification.html
        X_train, y_train, X_test, y_test = iterative_train_test_split(X_ds, y_ds, test_size=0.1)
        # convert the arrays back to dataframes, recreate the column headers and save the dataframes
        X_train_df = pd.DataFrame.from_records(X_train)
        X_train_df.columns = x_colnames
        y_train_df = pd.DataFrame.from_records(y_train)
        y_train_df.columns = y_colnames
        X_test_df = pd.DataFrame.from_records(X_test)
        X_test_df.columns = x_colnames
        y_test_df = pd.DataFrame.from_records(y_test)
        y_test_df.columns = y_colnames
        # generate the filenames and prepend the relevant paths
        xtrain_fn = traintestdir + 'dataset{}_Xtrain.csv'.format(dataset_counter)
        X_train_df.to_csv(xtrain_fn, index=False, header=True, encoding='UTF-8')
        ytrain_fn = traintestdir + 'dataset{}_ytrain.csv'.format(dataset_counter)
        y_train_df.to_csv(ytrain_fn, index=False, header=True, encoding='UTF-8')
        xtest_fn = traintestdir + 'dataset{}_Xtest.csv'.format(dataset_counter)
        ytest_fn = traintestdir + 'dataset{}_ytest.csv'.format(dataset_counter)
        X_test_df.to_csv(xtest_fn, index=False, header=True, encoding='UTF-8')
        y_test_df.to_csv(ytest_fn, index=False, header=True, encoding='UTF-8')
        # calculate some label combination vs. observation properties - how many examples available per label combination
        # (raw) and then generate the label allocation results, i.e.: how many observations of which combinations do we have
        # in train/test? The code below uses a function described in the scikit-multilearn userguide documented here:
        # http://scikit.ml/stratification.html
        label_alloc_diagnostics = pd.DataFrame({
            'raw': Counter(
                combination for row in get_combination_wise_output_matrix(y_ds, order=2) for combination in row),
            'train': Counter(
                str(combination) for row in get_combination_wise_output_matrix(y_train, order=2) for combination in
                row),
            'test': Counter(
                str(combination) for row in get_combination_wise_output_matrix(y_test, order=2) for combination in row)
        }).T.fillna(0.0)
        # generate a short status print of the dataset sizes
        print("Test data size: {}".format(y_test_df.shape[0]))
        print("Train data size: {}".format(y_train_df.shape[0]))
        # save this diagnostic data to the same directory
        diag_fn = traintestdir + 'dataset{}_split_diagnostics.csv'.format(dataset_counter)
        label_alloc_diagnostics.to_csv(diag_fn, index=False, header=True, encoding='UTF-8')
        # finally, increment counter
        dataset_counter += 1

    ############################################### DONE ###############################################################

    print("General preprocessing done now! For further preprocessing, check the individual model scripts!")
    ####################################################################################################################

    # if everything has worked out, return True
    return True


if __name__ == "__main__":  #
    # notify user that this is just a data-preprocessing module
    print("This module only provides data preprocessing functions, "
          "please import into another script for working with the ANES 2008 (Open Ended Coding Project)- data!")