"""This is code to read in the provided data from ANES 2008 (Open Ended Coding Project), split
the main dataset with excel-sheets of verbatims into subfiles and then match them with their (en-)codings.
After preliminary saving of the combined individual datasets, all data is collected into question-answer
combinations following Card/Smith 2016 and then saved. Finalized on: 05.04.2020"""

import os
import pandas as pd
import numpy as np


############################# DECLARATION OF HELPER-FUNCTIONS HERE #####################################################

def varname_generator(id_name, code_name, code_range):
    """Generates a list of variable names by cycling through from 1 to the maximum range set by the code_range
    variable and adding the integer suffix to the string contained in the code_name variable. This generated list is
    later used for dataset-subsetting and data selection.

    :param id_name: contains the name of the dataset row - identifier which is later used for merging datasets
    :param code_name: contains the name of the coding variable used in a specific dataset
    :param code_range: contains an integer, the function will generate a list of variable names
                       by cycling through from 1 to the maximum range set by this variable and adding
                       the integer suffix to the string contained in the code_name variable
    :return: a list of variable names as strings
    """
    # generate variable names according to the ones mentioned in the respective dataset coding report
    select_vars = [id_name]
    for i in range(1, (code_range + 1)):
        # creates a varname and adds an integer to it
        var_name = code_name + '{!s}'.format(i)
        select_vars.append(var_name)
    # finally, return the varname list
    return select_vars


def converter_func(input_row_list, mapping_dict, bin_vector_length=30):
    """Converts a list of codes pulled from a row of a codes-dataset in ANES 2008
    into a vector which contains 1s at the position defined in the code-mapping dict
    for each unique code assigned to the row in question. Performs a multi-label
    encoding by doing this.

    :param input_row_list: list of row values (no default), comes from a row from a codes-dataset from ANES 2008
    :param mapping_dict: dictionary which maps each unique code to a position in a binary vector
    :param bin_vector_length: length of output binary vector (has to be the same as the dictionary)
    :return: a binary vector which contains 1s at the position defined in the code-mapping dict
            for each unique code assigned to the row in question
    """
    # check if length of code positions is as long as the desired binary position vector
    if len(mapping_dict) != bin_vector_length:
        raise NameError("Error: mapping dict length != length of bin vector!")

    # generate a numpy array of zeros of length bin_vector_length
    binarized_vector = np.zeros(bin_vector_length)

    # only for crosscheck with xxxx: print input
    # print("Input original form: {}".format(input_row_list))

    # check if input is just a single integer
    if isinstance(input_row_list, (int, np.integer)):
        # case 1: input_row_list is just a single integer (only exceptions, only in optional datasets)

        # check if mapping dict is empty
        if len(mapping_dict) == 0:
            raise NameError("Error: mapping dict empty")
        else:
            # convert to string
            integer_element = input_row_list
            str_element = str(integer_element)
            # only proceed if element is not just an empty string
            if str_element.isspace() == False:
                # check if element has a mapping in mapping dict
                if str_element in mapping_dict:
                    # get mapping
                    element_mapping = mapping_dict[str_element]
                    # and use them to set a 1 at the index in the output binary
                    indexer = int(element_mapping) - 1  # because index starts at 0
                    binarized_vector[indexer] = 1
                else:
                    print(str_element)
                    raise NameError("Value unknown in mapping dict!")
    else:
        # case 2: input_row_list is a list (majority of cases, all datasets except the optional datasets)
        # check if input or mapping dict is empty
        if len(input_row_list) == 0 or len(mapping_dict) == 0:
            raise NameError("Error: input list empty / mapping dict empty")
        else:
            # check numbers (position markers) of input list
            for element in input_row_list:
                # convert to string
                str_element = str(element)
                # only proceed if element is not just an empty string
                if str_element.isspace() == False:
                    # check if element has a mapping in mapping dict
                    if str_element in mapping_dict:
                        # get mapping
                        element_mapping = mapping_dict[str_element]
                        # and use them to set a 1 at the index in the output binary
                        indexer = int(element_mapping) - 1  # because index starts at 0
                        binarized_vector[indexer] = 1
                    else:
                        print(str_element)
                        raise NameError("Value unknown in mapping dict!")
    # finally return the binarized vector
    binarized_vector = binarized_vector.astype(int)  # integerize the vector, only integers
    # only for crosscheck with xxx: print input
    # print("converted input: {}".format(binarized_vector))
    return binarized_vector


def transformer_function(path_to_df, list_of_vars_selection, mappings_dictionary, num_code_vars, path_to_output):
    """Transforms a codes-dataset from ANES 2008 coding project into a dataset with a multi-label encoding for each
    row of the original dataset. It takes in a row for a specific respondent - ID from the original data, isolates
    all (unique) codes assigned to this row and transforms these assigned codes into a binary vector containing
    1s at each position of the vector which has been mapped to the corresponding code using the mapping-dictionary,
    0s otherwise. This is done iteratively for the complete dataset. Afterwards, the new multi-label dataset is stored
    on the HDD, using the following structure for the dataset - variables: 1. caseID variable, 2. encoded binarized
    codes vector. The function also returns the encoded dataset for direct use.

    :param path_to_df: path to the original codes-dataset from ANES 2008
    :param list_of_vars_selection: code variables of interest in the specified datset
    :param mappings_dictionary: dictionary which assigns each code in the data a unique position
                                in the binary vector for the row
    :param num_code_vars: integer specifying how many positions the binary row vector has to have (= length of vector)
    :param path_to_output: path to the file where the encoded dataset has to be stored
    :return: encoded dataset as a pandas-dataframe
    """
    # shoot off status message
    print("[TRANSFORMER FUNCTION]: Starting transformation!")
    # load up the codes csv, subset to the variable of interest
    loader = pd.read_csv(path_to_df, sep=";")
    print("TRANSFORMER: orig row count ", loader.shape[0])
    loader = loader.loc[:, list_of_vars_selection]
    print("TRANSFORMER: subset row count ", loader.shape[0])
    # define a new dataframe which we are going to fill with the caseID, and a number of num_code_vars 0/1-columns
    # which represent markers for the unique codes used per verbatim
    list_of_keys = list(mappings_dictionary.keys())  # easier
    colnames_final_df = ["caseID"] + list_of_keys
    # arrange new dataframe
    new_df = pd.DataFrame(columns=colnames_final_df, index=range(0, loader.shape[0]))
    # overwrite the labels of the columns with the mapping dict keys
    new_df.columns = colnames_final_df
    # transform rows into a row with unique values, do this for all rows then build the new df with the same number
    # of rows as the original and num_code_vars-columns, fill with the binary 0 or 1s at the marker position (first,
    # second, etc.) of the original codes
    for index, row in loader.iterrows():  # old: range(0, loader.shape[0])
        # pull a row from the original dataframe and subset to the variables of interest
        old_row = row[list_of_vars_selection[1:]]
        # pull current row (case) ID from loader-dataframe
        old_row_caseID = row['ID']  # this variable is uniquely named "ID" in all codes files
        # old:  loader.loc[index, "ID"]
        # print("Old row values: ", old_row.values, " / type: ", type(old_row.values))
        # print("Row raw: ", old_row)
        # old: old_row = loader.loc[index, list_of_vars_selection[1:]]
        # get unique values from that row
        unique_values = list(set(old_row.values))
        # old: if isinstance(old_row, (int, np.integer)):
        #    unique_values = old_row
        # else:
        #    unique_values = list(set(old_row.values))
        # convert the list of unique vals into a binarized vector
        binarized_row = converter_func(unique_values, mappings_dictionary, bin_vector_length=num_code_vars)
        # concat caseID and binarized row vector to list for row
        new_row = [old_row_caseID] + binarized_row.tolist()
        # and finally push this into the new dataframe
        new_df.at[index, :] = new_row  # roll over all columns!
    # shoot off status message
    print("[TRANSFORMER FUNCTION]: Transformation finished!")
    # return and save the final new dataframe
    try:
        new_df.to_csv(path_to_output, index=False, header=True, encoding='utf-8')
    except:
        print("Error: could not save new dataframe to specified file-path (maybe already opened by another program?)")
    return new_df


def verbatim_code_merger(verbatim_df, binarized_codes_df, output_path):
    """Merges a pandas-dataframe containing a verbatims - (excel-sheet) file from ANES 2008 with a pandas-dataframe
    containing the multi-label encoded codes assigned to each verbatim. This uses the special structure of the
    ANES 2008 verbatim and codes - files containing a unique caseID - identifier variable. Both datasets are merged
    using the caseID-variable. The function stores the merged dataset on HDD and also returns it for direct use.

    :param verbatim_df: pandas-dataframe containing the verbatims assigned to the codes in the codes-dataframe
    :param binarized_codes_df: pandas-dtaframe containing the codes assigned to the verbatims
    :param output_path: path to where the merged dataset will be stored on the HDD
    :return: merged dataset as pandas-dataframe
    """
    # merge the dataframe in the following order: caseID, verbatim, binary code columns
    # rename the second column (verbatim column) of the verbatim df to "verbatim" for later ease in merging
    verbatim_df.columns = ['caseID', 'verbatim']
    print("Verbatim df variables: ", verbatim_df.columns)
    print("Codes df variables: ", binarized_codes_df.columns)
    merged_dataframe = pd.merge(verbatim_df, binarized_codes_df, on='caseID')
    # save the dataframe to file
    try:
        merged_dataframe.to_csv(output_path, index=False, header=True, encoding='utf-8')
    except:
        print("Error: could not save new dataframe to specified file-path (maybe already opened by another program?)")
    return merged_dataframe


############################################ MAIN FUNCTION #############################################################

def get_anes_data(rawdata_dir, targetdir, targetdir_codes, targetdir_verbatims, optional_ds=False):
    """Loads the verbatim excel file provided with the ANES 2008 open-ended coding
    project data and generates the matched verbatim-code files for each of the
    10 datasets mentioned in Card/Smith 2015. Intermediate isolated code - files,
    matched code - verbatim files and the final datasets are saved in the "targetdir"-
    folders. Optionally, it also generates verbatim-code files for the occupation
    questions, which have not been used in Card/Smith 2015. The function returns
    a dataframe map containing the 10 final datasets with verbatims and their codes.

    :param rawdata_dir: path to the folder holding the raw data (excel file and subfolders with codes)
    :param targetdir: path to the folder which will hold the intermediate and final datasets
    :param targetdir_codes: path to the folder which will hold the intermediate datasets containing codes
    :param optional_ds: optional switch for also creating verbatim-codes datasets for the occupation/industry-questions
    :return: dataframe map containing the 10 final datasets with verbatims and their codes mentioned in Card/Smith 2015
    """
    # setup preliminaries
    input_path = rawdata_dir + "anes_timeseries_2008_openends_redacted_Dec2012Revision.xls"

    # 1. make new directory in the project root where we want to put the matched verbatim-codes datasets
    # 2. and a directory where we want to put the extracted verbatims
    # 3. and a directory where we want to put the extracted codes for the verbatims-files
    folders = [targetdir, targetdir_verbatims, targetdir_codes]
    for folder in folders:
        # create the mentioned folders if they do not exist
        if not os.path.exists(folder):
            os.makedirs(folder)

    # import verbatim excel-file for splitting
    xlsfile = pd.ExcelFile(input_path)
    sheet_names = xlsfile.sheet_names

    # split the verbatims in the openends_redacted-file into individual datasets and save
    # from that always extract first and last column, slap header (from the names) above,
    # save as new datasets (just relevant text), leave out the overview sheet up front
    counter = 0
    sheet_df_map = {}
    for current_name in sheet_names[1:]:
        # print status
        counter += 1
        print("Processing sheet (", counter, ") :", current_name)

        # load file, get variable names, select first and last variable (caseID and verbatim - question)
        sheet_df = pd.read_excel(input_path, sheet_name=current_name)
        colnames = sheet_df.columns.values
        select_vars = colnames[::len(colnames) - 1]
        print("Selected vars: ", select_vars)

        # save individual sheet to csv for documentation purposes
        sheet_df.to_csv(targetdir + current_name + ".csv", index=False, header=False, encoding='utf-8')

        # always extract the first and third column for our text learning purposes by sheet
        # drop the first two rows from the excels (descriptions, big header, etc)
        verbatim_extract_df = pd.DataFrame(sheet_df, columns=select_vars)
        verbatim_extract_df = verbatim_extract_df.iloc[1:]
        # and save (for manual inspection!)
        verbatim_extract_df.to_csv(targetdir_verbatims + current_name + "_verbatim.csv",
                                   index=False,
                                   header=True,
                                   encoding='utf-8')

        # add the df to the df map for later convenient use
        sheet_df_map[current_name] = verbatim_extract_df

    # now load these verbatims one by one, merge with codes, save to thematically split question-answer groups
    # (to keep comparable to Card/Smith) and collect them into a dataframe map (initialized here)
    return_df_map = {}

    ######################################### Dataset 1: General Election ##############################################
    # Reasons why McCain lost the general election (MCGEN_Code1 to MCGEN_Code13), Sheet: WhyElectLose
    # Reasons why Obama won the general election (OBGEN_Code1 to OBGEN_Code13), Sheet: WhyElectWin

    # get input data filename
    elect_outcome_codes = rawdata_dir + "anes_timeseries_2008_election_outcomes/" + \
                          "anes_timeseries_2008_election_outcomes_all_codes.csv"

    # we have to go through each dataset individually, isolate the codings, transform them into a binary 1x46 vector
    # (46 because we have 46 unique codes used in the dataset, see word coument on codes). Finally, we can then merge
    # them into one big dataset where all verbatims by question are stacked under each other

    # setup the code to position mapping dict here (should be valid for all 4 question-answer encodings):
    gen_elect_dict = {"1": 1, "2": 2, "3": 3, "5": 4, "7": 5, "8": 6, "9": 7, "10": 8, "11": 9, "12": 10, "13": 11,
                      "14": 12, "15": 13, "16": 14, "17": 15, "18": 16, "19": 17, "20": 18, "21": 19, "22": 20,
                      "23": 21, "24": 22, "25": 23, "26": 24, "27": 25, "28": 26, "29": 27, "30": 28, "31": 29,
                      "32": 30, "33": 31, "34": 32, "35": 33, "36": 34, "37": 35, "38": 36, "39": 37, "40": 38,
                      "41": 39, "42": 40, "94": 41, "95": 42, "96": 43, "97": 44, "98": 45, "99": 46}
    # unknown codes (not mentioned in code overview): 3, 18, 11
    # 1. Reasons why McCain lost the general election (MCGEN_Code1 to MCGEN_Code13)

    # setup desired output path
    mccain_output_path = targetdir_codes + "/mccain_eleclost.csv"
    # generate variable names for data extraction according to the ones mentioned in the respective dataset
    # coding report
    mccain_select_vars = varname_generator("ID", "MCGEN_Code", 13)
    # transform to binarized code dataframe and save
    mccain_elec_df = transformer_function(path_to_df=elect_outcome_codes,
                                          list_of_vars_selection=mccain_select_vars,
                                          mappings_dictionary=gen_elect_dict,
                                          num_code_vars=46,
                                          path_to_output=mccain_output_path)
    print("McCain orig ds rowcount (codes):  ", mccain_elec_df.shape[0])
    # match with verbatims and save
    vmccain_output_path = targetdir + "/mccain_verbatim_code_eleclost.csv"
    mccain_eleclost_df = verbatim_code_merger(verbatim_df=sheet_df_map['WhyElectLose'],
                                              binarized_codes_df=mccain_elec_df,
                                              output_path=vmccain_output_path)
    print("McCain merged ds rowcount (codes):  ", mccain_eleclost_df.shape[0])
    print("Finished WhyElectLose McCain")
    # 2. Reasons why Obama won the general election (OBGEN_Code1 to OBGEN_Code13)

    # setup desired output path
    obama_output_path = targetdir_codes + "/obama_elecwin.csv"

    # generate variable names for data extraction according to the ones mentioned in the respective
    # dataset coding report
    obama_select_vars = varname_generator("ID", "OBGEN_Code", 13)

    # transform to binarized code dataframe and save
    obama_elec_df = transformer_function(path_to_df=elect_outcome_codes,
                                         list_of_vars_selection=obama_select_vars,
                                         mappings_dictionary=gen_elect_dict,
                                         num_code_vars=46,
                                         path_to_output=obama_output_path)
    print("Obama orig ds rowcount (codes):  ", obama_elec_df.shape[0])
    # match with verbatims and save
    vobama_output_path = targetdir + "/obama_verbatim_code_elecwin.csv"
    obama_win_df = verbatim_code_merger(verbatim_df=sheet_df_map['WhyElectWin'],
                                        binarized_codes_df=obama_elec_df,
                                        output_path=vobama_output_path)
    print("Obama merged ds rowcount (codes):  ", obama_win_df.shape[0])
    print("Finished WhyElectWin Obama")

    # 3. Match the codes with the verbatims and stack the verbatims and codes for both into a dataset
    # (following Card/Smith)
    pieces = (obama_win_df, mccain_eleclost_df)
    general_election = pd.concat(pieces, ignore_index=True)
    return_df_map['Dataset_1'] = general_election  # save to df map (same name as in Card/Smith)

    # sanity check: is sum of individual df row numbers equal final df row number?
    rows_sum = obama_win_df.shape[0] + mccain_eleclost_df.shape[0]
    print("(Dataset 1) Sanity check on row sum: ",
          rows_sum == general_election.shape[0],
          "(Sum: ", general_election.shape[0],
          ")")
    genlec_output_path = targetdir + "/Elecwinlost_verbatim_codes.csv"
    general_election.to_csv(genlec_output_path, index=False, header=True, encoding='utf-8')

    ######################################### Dataset 2: Primary Election ##############################################
    # Reasons why Clinton lost the Democratic nomination (CLPRIM_Code1 to CLPRIM_Code13), Sheet: WhyNomLose
    # Reasons why Obama won the Democratic nomination (OBPRIM_Code1 to OBPRIM_Code13), Sheet: WhyNomWin

    # get input data filename
    nom_outcome_codes = rawdata_dir + "anes_timeseries_2008_election_outcomes/" + \
                        "anes_timeseries_2008_election_outcomes_all_codes.csv"

    # we have to go through each dataset individually, isolate the codings, transform them into a binary 1x46 vector
    # (46 because we have 46 unique codes used in the dataset, see word coument on codes). Finally, we can then merge
    # them into one big dataset where all verbatims by question are stacked under each other

    # setup the code to position mapping dict here (should be valid for all 4 question-answer encodings):
    noutcomes_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11,
                      "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "19": 18, "20": 19, "21": 20,
                      "22": 21,
                      "23": 22, "24": 23, "25": 24, "26": 25, "27": 26, "28": 27, "29": 28, "30": 29, "31": 30,
                      "32": 31,
                      "33": 32, "34": 33, "35": 34, "36": 35, "37": 36, "38": 37, "39": 38, "40": 39, "41": 40,
                      "42": 41,
                      "94": 42, "95": 43, "96": 44, "97": 45, "98": 46, "99": 47}
    # unknown codes (not mentioned in code overview): 3,4,11
    #  1. Why do you think Barack Obama won the Democratic nomination? (OBPRIM_Code1 to OBPRIM_Code13)

    # setup desired output path
    nobama_output_path = targetdir_codes + "/obama_nomwin.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    nobama_select_vars = varname_generator("ID", "OBPRIM_Code", 13)
    # transform to binarized code dataframe and save
    nobama_df = transformer_function(path_to_df=nom_outcome_codes,
                                     list_of_vars_selection=nobama_select_vars,
                                     mappings_dictionary=noutcomes_dict,
                                     num_code_vars=47,
                                     path_to_output=nobama_output_path)

    # match with verbatims and save
    vnobama_output_path = targetdir + "/obama_verbatim_code_nomwin.csv"
    obama_nomwin_df = verbatim_code_merger(verbatim_df=sheet_df_map['WhyNomWin'],
                                           binarized_codes_df=nobama_df,
                                           output_path=vnobama_output_path)
    print("Finished WhyNomWin Obama")

    # 2. Why do you think Hillary Clinton lost the Democratic nomination? (CLPRIM_Code1 to CLPRIM_Code13)

    # setup desired output path
    nclinton_output_path = targetdir_codes + "/clinton_nomlost.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    nclinton_select_vars = varname_generator("ID", "CLPRIM_Code", 13)
    # transform to binarized code dataframe and save
    nclinton_df = transformer_function(path_to_df=nom_outcome_codes,
                                       list_of_vars_selection=nclinton_select_vars,
                                       mappings_dictionary=noutcomes_dict,
                                       num_code_vars=47,
                                       path_to_output=nclinton_output_path)

    # match with verbatims and save
    vnclinton_output_path = targetdir + "/clinton_verbatim_code_nomlost.csv"
    clinton_nomlost_df = verbatim_code_merger(verbatim_df=sheet_df_map['WhyNomLose'],
                                              binarized_codes_df=nclinton_df,
                                              output_path=vnclinton_output_path)
    print("Finished WhyNomLose Clinton")

    # 3. Match the codes with the verbatims and stack the verbatims and codes for both into a dataset
    # (following Card/Smith)
    pieces = (obama_nomwin_df, clinton_nomlost_df)
    primary_election = pd.concat(pieces, ignore_index=True)
    return_df_map['Dataset_2'] = primary_election  # save to df map (same name as in Card/Smith)

    # sanity check: is sum of individual df row numbers equal final df row number?
    rows_sum = obama_nomwin_df.shape[0] + clinton_nomlost_df.shape[0]
    print("(Dataset 2) Sanity check on row sum: ",
          rows_sum == primary_election.shape[0],
          "(Sum: ", primary_election.shape[0],
          ")")
    pelec_output_path = targetdir + "/Nomwinlost_verbatim_codes.csv"
    primary_election.to_csv(pelec_output_path, index=False, header=True, encoding='utf-8')

    ######################################### Dataset 3: Party (Dis)likes ##############################################
    # Dislikes about the Republican Party (REPDL_Code1 to REPDL_Code13), Sheet: RptyDislik
    # Likes about the Republican Party (REPLI_Code1 to REPLI_Code13), Sheet: RptyLike
    # Dislikes about the Democratic Party (DEMDL_Code1 to DEMDL_Code13), Sheet: DptyDislik
    # Likes about the Democratic Party (DEMLI_Code1 to DEMLI_Code13), Sheet: DptyLike

    # get input data filename
    party_dlikes_codes = rawdata_dir + "anes_timeseries_2008_party_likes_and_dislikes/" + \
                         "anes_timeseries_2008_party_likes_and_dislikes_all_codes.csv"

    # we have to go through each dataset individually, isolate the codings, transform them into a binary 1x 41 vector
    # (41 because we have 41 unique codes used in the dataset, see word coument on codes). Finally, we can then merge
    # them into one big dataset where all verbatims by question are stacked under each other

    # setup the code to position mapping dict here (should be valid for all 4 question-answer encodings):
    pdlikes_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11,
                    "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20, "21": 21,
                    "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29, "30": 30, "31": 31,
                    "32": 32, "44": 33, "45": 34, "46": 35, "94": 36, "95": 37, "96": 38, "97": 39, "98": 40, "99": 41}
    # unknown codes (not mentioned in code overview): 19, 2, 9, 8, 17, 7
    # 1. Dislikes about the Republican Party (REPDL_Code1 to REPDL_Code13)

    # setup desired output path
    disl_output_path = targetdir_codes + "/repparty_dislikes.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    disl_select_vars = varname_generator("ID", "REPDL_Code", 13)
    # transform to binarized code dataframe and save
    rep_disl_df = transformer_function(path_to_df=party_dlikes_codes,
                                       list_of_vars_selection=disl_select_vars,
                                       mappings_dictionary=pdlikes_dict,
                                       num_code_vars=41,
                                       path_to_output=disl_output_path)

    # match with verbatims and save
    vrep_disl_output_path = targetdir + "/repparty_verbatim_dislikes.csv"
    rep_dlikes_df = verbatim_code_merger(verbatim_df=sheet_df_map['RptyDislik'],
                                         binarized_codes_df=rep_disl_df,
                                         output_path=vrep_disl_output_path)

    print("Finished RptyDislik: Republican Party dislikes")

    # 2. Likes about the Republican Party (REPLI_Code1 to REPLI_Code13) RptyLike

    # setup desired output path
    like_output_path = targetdir_codes + "/repparty_likes.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    like_select_vars = varname_generator("ID", "REPLI_Code", 13)
    # transform to binarized code dataframe and save
    rep_like_df = transformer_function(path_to_df=party_dlikes_codes,
                                       list_of_vars_selection=like_select_vars,
                                       mappings_dictionary=pdlikes_dict,
                                       num_code_vars=41,
                                       path_to_output=like_output_path)

    # match with verbatims and save
    vrep_like_output_path = targetdir + "/repparty_verbatim_likes.csv"
    rep_likes_df = verbatim_code_merger(verbatim_df=sheet_df_map['RptyLike'],
                                        binarized_codes_df=rep_like_df,
                                        output_path=vrep_like_output_path)

    print("Finished RptyLike: Republican Party likes")

    # 3. Dislikes about the Democratic Party (DEMDL_Code1 to DEMDL_Code13) DptyDislik

    # setup desired output path
    ddislik_output_path = targetdir_codes + "/demparty_dislikes.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    ddislik_select_vars = varname_generator("ID", "DEMDL_Code", 13)
    # transform to binarized code dataframe and save
    dem_dislik_df = transformer_function(path_to_df=party_dlikes_codes,
                                         list_of_vars_selection=ddislik_select_vars,
                                         mappings_dictionary=pdlikes_dict,
                                         num_code_vars=41,
                                         path_to_output=ddislik_output_path)

    # match with verbatims and save
    vdem_dislik_output_path = targetdir + "/demparty_verbatim_dislikes.csv"
    dem_dlikes_df = verbatim_code_merger(verbatim_df=sheet_df_map['DptyDislik'],
                                         binarized_codes_df=dem_dislik_df,
                                         output_path=vdem_dislik_output_path)

    print("Finished DptyDislik: Democratic Party dislikes")

    # 4. Likes about the Democratic Party (DEMLI_Code1 to DEMLI_Code13) DptyLike

    # setup desired output path
    dem_like_output_path = targetdir_codes + "/demparty_likes.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    dem_like_select_vars = varname_generator("ID", "DEMLI_Code", 13)
    # transform to binarized code dataframe and save
    dem_like_df = transformer_function(path_to_df=party_dlikes_codes,
                                       list_of_vars_selection=dem_like_select_vars,
                                       mappings_dictionary=pdlikes_dict,
                                       num_code_vars=41,
                                       path_to_output=dem_like_output_path)

    # match with verbatims and save
    vdemlike_output_path = targetdir + "/demparty_verbatim_likes.csv"
    dem_likes_df = verbatim_code_merger(verbatim_df=sheet_df_map['DptyLike'],
                                        binarized_codes_df=dem_like_df,
                                        output_path=vdemlike_output_path)

    print("Finished DptyLike: Democratic Party likes")

    # 5. Match the codes with the verbatims and stack the verbatims and codes for both into a dataset
    # (following Card/Smith)
    pieces = (dem_likes_df, rep_likes_df, dem_dlikes_df, rep_dlikes_df)
    party_dlikes = pd.concat(pieces, ignore_index=True)
    return_df_map['Dataset_3'] = party_dlikes  # save to df map (same name as in Card/Smith)

    # sanity check: is sum of individual df row numbers equal final df row number?
    rows_sum = dem_likes_df.shape[0] + rep_likes_df.shape[0] + dem_dlikes_df.shape[0] + rep_dlikes_df.shape[0]
    print("(Dataset 3) Sanity check on row sum: ",
          rows_sum == party_dlikes.shape[0],
          "(Sum: ", party_dlikes.shape[0],
          ")")
    output_path = targetdir + "/Party_dlikes_verbatim_codes.csv"
    party_dlikes.to_csv(output_path, index=False, header=True, encoding='utf-8')

    ######################################### Dataset 4: Person (Dis)likes #############################################
    # Reasons to vote against John McCain (MCCDL_Code1 to MCCDL_Code21), Sheet: RcandDislik
    # Reasons to vote for John McCain (MCCLI_Code1 to MCCLI_Code21), Sheet: RcandLike
    # Reasons to vote against Barack Obama (OBADL_Code1 to OBADL_Code21), Sheet: DcandDislik
    # Reasons to vote for Barack Obama (OBALI_Code1 to OBALI_Code21), Sheet: DcandLike

    # get input data filename
    lkdk_cand_codes = rawdata_dir + "anes_timeseries_2008_candidate_likes_and_dislikes/" + \
                      "anes_timeseries_2008_candidate_likes_and_dislikes_all_codes.csv"

    # we have to go through each dataset individually, isolate the codings, transform them into a binary 1x39 vector
    # (39 because we have 39 unique codes used in the dataset, see word coument on codes). Finally, we can then merge
    # them into one big dataset where all verbatims by question are stacked under each other

    # setup the code to position mapping dict here (should be valid for all 4 question-answer encodings):
    cand_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11,
                 "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20, "21": 21,
                 "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29, "30": 30, "31": 31,
                 "32": 32, "33": 33, "43": 34, "94": 35, "95": 36, "96": 37, "97": 38, "98": 39, "99": 40}

    # 1. DcandLike: Reasons to vote for Barack Obama (OBALI_Code1 to OBALI_Code21)

    # setup desired output path
    fobama_output_path = targetdir_codes + "/obama_likes.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    fobama_select_vars = varname_generator("ID", "OBALI_Code", 21)
    # transform to binarized code dataframe and save
    vfor_obama_df = transformer_function(path_to_df=lkdk_cand_codes,
                                         list_of_vars_selection=fobama_select_vars,
                                         mappings_dictionary=cand_dict,
                                         num_code_vars=40,
                                         path_to_output=fobama_output_path)

    # match with verbatims and save
    vfobama_output_path = targetdir + "/obama_verbatim_code_likes.csv"
    obama_likes_df = verbatim_code_merger(verbatim_df=sheet_df_map['DcandLike'],
                                          binarized_codes_df=vfor_obama_df,
                                          output_path=vfobama_output_path)
    print("Finished DcandLike Obama")

    # 2. DcanddLike: Reasons to vote against Barack Obama (OBADL_Code1 to OBADL_Code21)

    # setup desired output path
    vaobama_output_path = targetdir_codes + "/obama_dislikes.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    vaobama_select_vars = varname_generator("ID", "OBADL_Code", 21)
    # transform to binarized code dataframe and save
    vagainst_obama_df = transformer_function(path_to_df=lkdk_cand_codes,
                                             list_of_vars_selection=vaobama_select_vars,
                                             mappings_dictionary=cand_dict,
                                             num_code_vars=40,
                                             path_to_output=vaobama_output_path)
    # match with verbatims and save
    vaobama_output_path = targetdir + "/obama_verbatim_code_dislikes.csv"
    obama_dislikes_df = verbatim_code_merger(verbatim_df=sheet_df_map['DcandDislik'],
                                             binarized_codes_df=vagainst_obama_df,
                                             output_path=vaobama_output_path)
    print("Finished DcandDislik Obama")

    # 3. Rcandlike: Reasons to vote for John McCain (MCCLI_Code1 to MCCLI_Code21)

    # setup desired output path
    fmccain_output_path = targetdir_codes + "/mccain_likes.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    fmccain_select_vars = varname_generator("ID", "MCCLI_Code", 21)
    # transform to binarized code dataframe and save
    vfor_maccain_df = transformer_function(path_to_df=lkdk_cand_codes,
                                           list_of_vars_selection=fmccain_select_vars,
                                           mappings_dictionary=cand_dict,
                                           num_code_vars=40,
                                           path_to_output=fmccain_output_path)
    # match with verbatims and save
    vformccain_output_path = targetdir + "/mccain_verbatim_code_likes.csv"
    mccain_likes_df = verbatim_code_merger(verbatim_df=sheet_df_map['RcandLike'],
                                           binarized_codes_df=vfor_maccain_df,
                                           output_path=vformccain_output_path)
    print("Finished RcandLike McCain")

    # 4. Rcanddlike: Reasons to vote against John McCain (MCCDL_Code1 to MCCDL_Code21)

    # setup desired output path
    vamccain_output_path = targetdir_codes + "/mccain_dislikes.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    vamccain_select_vars = varname_generator("ID", "MCCDL_Code", 21)
    # transform to binarized code dataframe and save
    vagainst_mccain_df = transformer_function(path_to_df=lkdk_cand_codes,
                                              list_of_vars_selection=vamccain_select_vars,
                                              mappings_dictionary=cand_dict,
                                              num_code_vars=40,
                                              path_to_output=vamccain_output_path)
    # match with verbatims and save
    vamccain_output_path = targetdir + "/mccain_verbatim_dislikes.csv"
    mccain_dislikes_df = verbatim_code_merger(verbatim_df=sheet_df_map['RcandDislik'],
                                              binarized_codes_df=vagainst_mccain_df,
                                              output_path=vamccain_output_path)
    print("Finished RcandDislik McCain")

    # 5. Match the codes with the verbatims and stack the verbatims and codes for both into a dataset
    # (following Card/Smith)
    pieces = (obama_likes_df, obama_dislikes_df, mccain_likes_df, mccain_dislikes_df)
    person_dlikes_df = pd.concat(pieces, ignore_index=True)
    return_df_map['Dataset_4'] = person_dlikes_df  # save to df map (same name as in Card/Smith)

    # sanity check: is sum of individual df row numbers equal final df row number?
    rows_sum = mccain_likes_df.shape[0] + obama_likes_df.shape[0] + \
               mccain_dislikes_df.shape[0] + obama_dislikes_df.shape[0]
    print("(Dataset 4) Sanity check on row sum: ",
          rows_sum == person_dlikes_df.shape[0],
          "(Sum: ",
          person_dlikes_df.shape[0],
          ")")
    output_path = targetdir + "/Cand_like_dislike_verbatim_codes.csv"
    person_dlikes_df.to_csv(output_path, index=False, header=True, encoding='utf-8')

    ############################################## Dataset 5: Terrorists ###############################################
    # Terrorists’ attacks (Terr1 to Terr11), Sheet: DHSsept11

    # get input data filename
    terrorists_codes = rawdata_dir + "anes_timeseries_2008_terrorists/" + \
                       "Terrorists - All codes.csv"

    # we have to go through each dataset individually, isolate the codings, transform them into a binary 1x30 vector
    # (30 because we have 30 unique codes used in the dataset, see word coument on codes). Finally, we can then merge
    # them into one big dataset where all verbatims by question are stacked under each other

    # setup the code to position mapping dict here:
    terrorist_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11,
                      "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20,
                      "21": 21, "22": 22, "23": 23, "24": 24, "25": 25, "91": 26, "96": 27, "97": 28, "98": 29,
                      "99": 30}

    # unknown codes (not mentioned in the code overview): none

    # also the master code variable description is wrong! it is not: Terrorists’ attacks (Terr1 to Terr11) BUT:
    # TER_Code1;TER_Code2;TER_Code3;TER_Code4;TER_Code5;TER_Code6;TER_Code7;TER_Code8;TER_Code9;TER_Code10;TER_Code11;

    # setup desired output path
    tcodes_output_path = targetdir_codes + "/terrorists.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    terr_select_vars = varname_generator("ID", "TER_Code", 11)
    # transform to binarized code dataframe and save
    terr_df = transformer_function(path_to_df=terrorists_codes,
                                   list_of_vars_selection=terr_select_vars,
                                   mappings_dictionary=terrorist_dict,
                                   num_code_vars=30,
                                   path_to_output=tcodes_output_path)
    # match with verbatims and save
    vterr_output_path = targetdir + "/terrrorists_verbatim_code.csv"
    terrorists_df = verbatim_code_merger(verbatim_df=sheet_df_map['DHSsept11'],
                                         binarized_codes_df=terr_df,
                                         output_path=vterr_output_path)

    print("Finished DHSsept11: Terrorists question on 9/11")

    # Collect the dataset into the final dataset map and save into a dataset (following Card/Smith)
    return_df_map['Dataset_5'] = terrorists_df  # save to df map (same name as in Card/Smith)

    # sanity check: is total of original codes-df row numbers equal final df row number?
    checktotal_orig_df = pd.read_csv(terrorists_codes, sep=";")
    rows_sum = checktotal_orig_df.shape[0]
    print("(Dataset 5) Sanity check on row sum: ",
          rows_sum == terrorists_df.shape[0],
          "(Sum: ",
          terrorists_df.shape[0],
          ")")
    output_path = targetdir + "/terrorists_verbatim_codes.csv"
    terrorists_df.to_csv(output_path, index=False, header=True, encoding='utf-8')

    ############################################## Dataset 6: Important Issues #########################################
    # Most Important Political Problem:
    #	MIPPOL1_Code1 to MIPPOL1_Code8
    #	MIPPOL1_Substantive1 to MIPPOL1_Substantive8, Sheet: MIPpolit1
    # Second Most Important Political Problem:
    #	MIPPOL2_Code1 to MIPPOL2_Code8
    #	MIPPOL2_Substantive1 to MIPPOL2_Substantive8, Sheet: MIPpolit2
    # Most Important Election Issue:
    #	MIIELE1_Code1 to MIIELE1_Code8
    #	MIIELE1_Substantive1 to MIIELE1_Substantive8, Sheet: MIPpers1
    # Second Most Important Election Issue:
    #	MIIELE2_Code1 to MIIELE2_Code8
    #	MIIELE2_Substantive1 to MIIELE2_Substantive8, Sheet: MIPpers2

    # get input data filename
    important_codes = rawdata_dir + "anes_timeseries_2008_most_important_problem/" + \
                      "Most Important Problem - All codes.csv"

    # we have to go through each dataset individually, isolate the codings, transform them into a binary 1x77 vector
    # (77 because we have 77 unique codes used in the dataset, see word coument on codes). Finally, we can then merge
    # them into one big dataset where all verbatims by question are stacked under each other

    # setup the code to position mapping dict here:
    mip_issues_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11,
                       "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20,
                       "21": 21, "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29,
                       "30": 30, "31": 31, "32": 32, "33": 33, "34": 34, "35": 35, "36": 36, "37": 37, "38": 38,
                       "39": 39, "40": 40, "41": 41, "42": 42, "43": 43, "44": 44, "45": 45, "46": 46, "47": 47,
                       "48": 48, "49": 49, "50": 50, "51": 51, "52": 52, "53": 53, "54": 54, "55": 55, "56": 56,
                       "57": 57, "58": 58, "59": 59, "60": 60, "61": 61, "62": 62, "63": 63, "64": 64, "65": 65,
                       "66": 66, "67": 67, "68": 68, "69": 69, "70": 70, "71": 71, "91": 72, "95": 73, "96": 74,
                       "97": 75, "98": 76, "99": 77}

    # unknown codes (not mentioned in the code overview): none
    # 1. Most Important Political Problem MIPPOL1_Code1 to MIPPOL1_Code8 and
    # MIPPOL1_Substantive1 to MIPPOL1_Substantive8

    # setup desired output path
    mip1_output_path = targetdir_codes + "/mip1_political.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    mip1_first_select_vars = varname_generator("ID", "MIPPOL1_Code", 8)
    mip1_second_select_vars = varname_generator("ID", "MIPPOL1_Substantive", 8)
    mip1_second_select_vars = mip1_second_select_vars[1:]  # throw away the second "ID" variable reference
    mip1_vars_selected_overall = mip1_first_select_vars + mip1_second_select_vars  # concat the two lists
    # transform to binarized code dataframe and save
    mip1_df = transformer_function(path_to_df=important_codes,
                                   list_of_vars_selection=mip1_vars_selected_overall,
                                   mappings_dictionary=mip_issues_dict,
                                   num_code_vars=77,
                                   path_to_output=mip1_output_path)
    # match with verbatims and save
    vmip1_output_path = targetdir + "/mip1_political_verbatim_code.csv"
    mip1_political = verbatim_code_merger(verbatim_df=sheet_df_map['MIPpolit1'],
                                          binarized_codes_df=mip1_df,
                                          output_path=vmip1_output_path)
    print("Finished MIPpolit1: Most important political issue question")
    # 2. Second Most Important Political Problem: MIPPOL2_Code1 to MIPPOL2_Code8 and MIPPOL2_Substantive1 to
    # MIPPOL2_Substantive8

    # setup desired output path
    mip2_output_path = targetdir_codes + "/mip2_political.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    mip2_first_select_vars = varname_generator("ID", "MIPPOL2_Code", 8)
    mip2_second_select_vars = varname_generator("ID", "MIPPOL2_Substantive", 8)
    mip2_second_select_vars = mip2_second_select_vars[1:]  # throw away the second "ID" variable reference
    mip2_vars_selected_overall = mip2_first_select_vars + mip2_second_select_vars  # concat the two lists
    # transform to binarized code dataframe and save
    mip2_df = transformer_function(path_to_df=important_codes,
                                   list_of_vars_selection=mip2_vars_selected_overall,
                                   mappings_dictionary=mip_issues_dict,
                                   num_code_vars=77,
                                   path_to_output=mip2_output_path)
    # match with verbatims and save
    vmip2_output_path = targetdir + "/mip2_political_verbatim_code.csv"
    mip2_political = verbatim_code_merger(verbatim_df=sheet_df_map['MIPpolit2'],
                                          binarized_codes_df=mip2_df,
                                          output_path=vmip2_output_path)
    print("Finished MIPpolit2: Second most important political issue question")
    # 3. Most Important Election Issue: MIIELE1_Code1 to MIIELE1_Code8 and MIIELE1_Substantive1 to MIIELE1_Substantive8

    # setup desired output path
    pip1_output_path = targetdir_codes + "/mielect1_political.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    pip1_first_select_vars = varname_generator("ID", "MIIELE1_Code", 8)
    pip1_second_select_vars = varname_generator("ID", "MIIELE1_Substantive", 8)
    pip1_second_select_vars = pip1_second_select_vars[1:]  # throw away the second "ID" variable reference
    pip1_vars_selected_overall = pip1_first_select_vars + pip1_second_select_vars  # concat the two lists
    # transform to binarized code dataframe and save
    pip1_df = transformer_function(path_to_df=important_codes,
                                   list_of_vars_selection=pip1_vars_selected_overall,
                                   mappings_dictionary=mip_issues_dict,
                                   num_code_vars=77,
                                   path_to_output=pip1_output_path)
    # match with verbatims and save
    vpip1_output_path = targetdir + "/mielect1_political_verbatim_code.csv"
    mielect1_political = verbatim_code_merger(verbatim_df=sheet_df_map['MIPpers1'],
                                              binarized_codes_df=pip1_df,
                                              output_path=vpip1_output_path)
    print("Finished MIPpers1: Most important personal topic in this election")
    # 4. Second Most Important Election Issue: MIIELE2_Code1 to MIIELE2_Code8 and  MIIELE2_Substantive1 to
    # MIIELE2_Substantive8

    # setup desired output path
    pip2_output_path = targetdir_codes + "/mielect2_political.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    pip2_first_select_vars = varname_generator("ID", "MIIELE2_Code", 8)
    pip2_second_select_vars = varname_generator("ID", "MIIELE2_Substantive", 8)
    pip2_second_select_vars = pip2_second_select_vars[1:]  # throw away the second "ID" variable reference
    pip2_vars_selected_overall = pip2_first_select_vars + pip2_second_select_vars  # concat the two lists
    # transform to binarized code dataframe and save
    pip2_df = transformer_function(path_to_df=important_codes,
                                   list_of_vars_selection=pip2_vars_selected_overall,
                                   mappings_dictionary=mip_issues_dict,
                                   num_code_vars=77,
                                   path_to_output=output_path)
    # match with verbatims and save
    vpip2_output_path = targetdir + "/mielect2_political_verbatim_code.csv"
    mielect2_political = verbatim_code_merger(verbatim_df=sheet_df_map['MIPpers2'],
                                              binarized_codes_df=pip2_df,
                                              output_path=vpip2_output_path)

    print("Finished MIPpers2: Second most important personal topic in this election")
    # 5. Match the codes with the verbatims and stack the verbatims and codes for both
    # into a dataset (following Card/Smith)

    pieces = (mielect1_political, mielect2_political, mip1_political, mip2_political)
    important_issues_df = pd.concat(pieces, ignore_index=True)
    return_df_map['Dataset_6'] = important_issues_df  # save to df map (same name as in Card/Smith)
    # sanity check: is sum of individual df row numbers equal final df row number?
    rows_sum = mielect1_political.shape[0] + mielect2_political.shape[0] + \
               mip1_political.shape[0] + mip2_political.shape[0]
    print("(Dataset 6) Sanity check on row sum: ",
          rows_sum == important_issues_df.shape[0],
          "(Sum: ",
          important_issues_df.shape[0],
          ")")
    output_path = targetdir + "/Important_Issues_verbatim_codes.csv"
    important_issues_df.to_csv(output_path, index=False, header=True, encoding='utf-8')

    ###################################### Dataset 7: Political Knowledge: Brown #######################################
    # Gordon Brown Office Recognition Codes (BROWN_Code1 to BROWN_Code5), Sheet: OfcBrown
    # get input data filename
    brown_ofrc_codes = rawdata_dir + "ANES2008TS_OfficeRecognition/" + \
                       "Political knowledge - All codes2.csv"

    # we have to go through each dataset individually, isolate the codings, transform them into a binary 1 x 16 vector
    # (16 because we have 16 unique codes used in the dataset, see word coument on codes). Finally, we can then merge
    # them into one big dataset where all verbatims by question are stacked under each other

    # setup the code to position mapping dict here:
    brown_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "11": 5, "12": 6, "13": 7, "14": 8, "21": 9, "23": 10, "24": 11,
                  "95": 12, "96": 13, "97": 14, "98": 15, "99": 16}
    # unknown codes (not mentioned in the code overview): none

    # setup desired output path
    brown_output_path = targetdir_codes + "/brown_ofrc.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    brown_select_vars = varname_generator("ID", "BROWN_Code", 5)
    # transform to binarized code dataframe and save
    brown_df = transformer_function(path_to_df=brown_ofrc_codes,
                                    list_of_vars_selection=brown_select_vars,
                                    mappings_dictionary=brown_dict,
                                    num_code_vars=16,
                                    path_to_output=brown_output_path)
    # match with verbatims and save
    vbrown_output_path = targetdir + "/brown_ofrc_verbatim_code.csv"
    brown_ofrc_df = verbatim_code_merger(verbatim_df=sheet_df_map['OfcBrown'],
                                         binarized_codes_df=brown_df,
                                         output_path=vbrown_output_path)
    # Collect the dataset into the final dataset map and save into a dataset (following Card/Smith)

    brown_checktotal_orig = pd.read_csv(brown_ofrc_codes, sep=";")
    return_df_map['Dataset_7'] = brown_ofrc_df  # save to df map (same name as in Card/Smith)
    # sanity check: is sum of individual df row numbers equal final df row number?
    rows_sum = brown_checktotal_orig.shape[0]
    print("(Dataset 7) Sanity check on row sum: ",
          rows_sum == brown_ofrc_df.shape[0],
          "(Sum: ",
          brown_ofrc_df.shape[0],
          ")")
    output_path = targetdir + "/brown_ofrc_verbatim_codes.csv"
    brown_ofrc_df.to_csv(output_path, index=False, header=True, encoding='utf-8')
    print("Finished OfcBrown: Office recognition question Gordon Brown")

    ###################################### Dataset 8: Political Knowledge: Cheney ######################################
    # Dick Cheney Office Recognition Codes (CHENEY_Code1 to CHENEY_Code5), Sheet: OfcCheney

    # get input data filename
    cheney_ofrc_codes = rawdata_dir + "ANES2008TS_OfficeRecognition/" + \
                        "Political knowledge - All codes2.csv"

    # we have to go through each dataset individually, isolate the codings, transform them into a binary 1 x 14 vector
    # (14 because we have 14 unique codes used in the dataset, see word coument on codes). Finally, we can then merge
    # them into one big dataset where all verbatims by question are stacked under each other

    # setup the code to position mapping dict here:
    cheney_dict = {"1": 1, "3": 2, "4": 3, "11": 4, "12": 5, "21": 6, "22": 7, "23": 8, "24": 9, "95": 10, "96": 11,
                   "97": 12, "98": 13, "99": 14}

    # unknown codes (not mentioned in the code overview): none

    # setup desired output path
    chen_output_path = targetdir_codes + "/cheney_ofrc.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    chen_select_vars = varname_generator("ID", "CHENEY_Code", 5)
    # transform to binarized code dataframe and save
    cheney_df = transformer_function(path_to_df=cheney_ofrc_codes,
                                     list_of_vars_selection=chen_select_vars,
                                     mappings_dictionary=cheney_dict,
                                     num_code_vars=14,
                                     path_to_output=chen_output_path)
    # match with verbatims and save
    vcheney_output_path = targetdir + "/cheney_ofrc_verbatim_code.csv"
    cheney_ofrc_df = verbatim_code_merger(verbatim_df=sheet_df_map['OfcCheney'],
                                          binarized_codes_df=cheney_df,
                                          output_path=vcheney_output_path)
    # Collect the dataset into the final dataset map and save into a dataset (following Card/Smith)

    chen_checktotal_orig = pd.read_csv(cheney_ofrc_codes, sep=";")
    return_df_map['Dataset_8'] = cheney_ofrc_df  # save to df map (same name as in Card/Smith)
    # sanity check: is sum of individual df row numbers equal final df row number?
    rows_sum = chen_checktotal_orig.shape[0]
    print("(Dataset 8) Sanity check on row sum: ",
          rows_sum == cheney_ofrc_df.shape[0],
          "(Sum: ",
          cheney_ofrc_df.shape[0],
          ")")
    output_path = targetdir_codes + "/cheney_ofrc_verbatim_codes.csv"
    cheney_ofrc_df.to_csv(output_path, index=False, header=True, encoding='utf-8')
    print("Finished OfcCheney: Office recognition question Dick Cheney")

    ###################################### Dataset 9: Political Knowledge: Pelosi ######################################
    # Nancy Pelosi Office Recognition Codes (PELOSI_Code1 to PELOSI_Code5), Sheet: OfcPelosi

    # get input data filename
    pelosi_ofrc_codes = rawdata_dir + "ANES2008TS_OfficeRecognition/" + \
                        "Political knowledge - All codes2.csv"

    # we have to go through each dataset individually, isolate the codings, transform them into a binary 1 x 17 vector
    # (17 because we have 17 unique codes used in the dataset, see word coument on codes). Finally, we can then merge
    # them into one big dataset where all verbatims by question are stacked under each other

    # setup the code to position mapping dict here:
    pelosi_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "11": 5, "12": 6, "15": 7, "16": 8, "21": 9, "22": 10, "23": 11,
                   "24": 12, "95": 13, "96": 14, "97": 15, "98": 16, "99": 17}
    # unknown codes (not mentioned in the code overview): none

    # setup desired output path
    pel_output_path = targetdir_codes + "/pelosi_ofrc.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    pel_select_vars = varname_generator("ID", "PELOSI_Code", 5)
    # transform to binarized code dataframe and save
    pelosi_df = transformer_function(path_to_df=pelosi_ofrc_codes,
                                     list_of_vars_selection=pel_select_vars,
                                     mappings_dictionary=pelosi_dict,
                                     num_code_vars=17,
                                     path_to_output=pel_output_path)
    # match with verbatims and save
    vpel_output_path = targetdir + "/pelosi_ofrc_verbatim_code.csv"
    pelosi_ofrc_df = verbatim_code_merger(verbatim_df=sheet_df_map['OfcPelosi'],
                                          binarized_codes_df=pelosi_df,
                                          output_path=vpel_output_path)
    # Collect the dataset into the final dataset map and save into a dataset (following Card/Smith)

    pel_checktotal_orig = pd.read_csv(pelosi_ofrc_codes, sep=";")
    return_df_map['Dataset_9'] = pelosi_ofrc_df  # save to df map (same name as in Card/Smith)
    # sanity check: is sum of individual df row numbers equal final df row number?
    rows_sum = pel_checktotal_orig.shape[0]
    print("(Dataset 9) Sanity check on row sum: ",
          rows_sum == pelosi_ofrc_df.shape[0],
          "(Sum: ",
          pelosi_ofrc_df.shape[0],
          ")")
    output_path = targetdir + "/pelosi_ofrc_verbatim_codes.csv"
    pelosi_ofrc_df.to_csv(output_path, index=False, header=True, encoding='utf-8')
    print("Finished OfcPelosi: Office recognition question Nancy Pelosi")

    ###################################### Dataset 10: Political Knowledge: Roberts ####################################
    # John Roberts Office Recognition Codes (ROBERTS_Code1 to ROBERTS_Code5), Sheet: OfcRoberts

    # get input data filename
    roberts_ofrc_codes = rawdata_dir + "ANES2008TS_OfficeRecognition/" + \
                         "Political knowledge - All codes2.csv"

    # we have to go through each dataset individually, isolate the codings, transform them into a binary 1 x 17 vector
    # (17 because we have 17 unique codes used in the dataset, see word coument on codes). Finally, we can then merge
    # them into one big dataset where all verbatims by question are stacked under each other

    # setup the code to position mapping dict here:
    roberts_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "11": 5, "12": 6, "13": 7, "14": 8, "21": 9, "23": 10, "24": 11,
                    "95": 12, "96": 13, "97": 14, "98": 15, "99": 16}

    # unknown codes (not mentioned in the code overview): none

    # setup desired output path
    rob_output_path = targetdir_codes + "/roberts_ofrc.csv"
    # generate variable names according to the ones mentioned in the respective dataset coding report
    rob_select_vars = varname_generator("ID", "ROBERTS_Code", 5)
    # transform to binarized code dataframe and save
    roberts_df = transformer_function(path_to_df=roberts_ofrc_codes,
                                      list_of_vars_selection=rob_select_vars,
                                      mappings_dictionary=roberts_dict,
                                      num_code_vars=16,
                                      path_to_output=rob_output_path)
    # match with verbatims and save
    vrob_output_path = targetdir + "/roberts_ofrc_verbatim_code.csv"
    roberts_ofrc_df = verbatim_code_merger(verbatim_df=sheet_df_map['OfcRoberts'],
                                           binarized_codes_df=roberts_df,
                                           output_path=vrob_output_path)

    # Collect the dataset into the final dataset map and save into a dataset (following Card/Smith)
    rob_checktotal_orig = pd.read_csv(roberts_ofrc_codes, sep=";")
    return_df_map['Dataset_10'] = roberts_ofrc_df  # save to df map (same name as in Card/Smith)
    # sanity check: is sum of individual df row numbers equal final df row number?
    rows_sum = rob_checktotal_orig.shape[0]
    print("(Dataset 10) Sanity check on row sum: ",
          rows_sum == roberts_ofrc_df.shape[0],
          "(Sum: ",
          roberts_ofrc_df.shape[0],
          ")")
    output_path = targetdir + "/roberts_ofrc_verbatim_codes.csv"
    roberts_ofrc_df.to_csv(output_path, index=False, header=True, encoding='utf-8')
    print("Finished OfcRoberts: Office recognition question John Roberts")

    ########################################### DATASETS CARD/ SMITH DONE ##############################################
    # Additionally: convert and merge the datasets for occupation and industry (for checks by interested persons or
    # raw data providers)
    if optional_ds == True:
        print("Generating optional datasets: occupation and industry")
        # A. Occupation codes and verbatims (past/present)

        # get input data filename
        occupation_codes = rawdata_dir + "anes_timeseries_2008_occupation_industry/" + \
                           "Occupation and Industry codes.csv"

        # we have to go through each dataset individually, isolate the codings, transform them into a binary 1 x 101
        # vector (101 because we have 101 unique codes used in the dataset, see word coument on codes).
        # Finally, we can then merge them into one big dataset where all verbatims by question are stacked
        # under each other

        # setup the code to position mapping dict here:
        occupations_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11,
                            "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20,
                            "21": 21, "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29,
                            "30": 30, "31": 31, "32": 32, "33": 33, "34": 34, "35": 35, "36": 36, "37": 37, "38": 38,
                            "39": 39, "40": 40, "41": 41, "42": 42, "43": 43, "44": 44, "45": 45, "46": 46, "47": 47,
                            "48": 48, "49": 49, "50": 50, "51": 51, "52": 52, "53": 53, "54": 54, "55": 55, "56": 56,
                            "57": 57, "58": 58, "59": 59, "60": 60, "61": 61, "62": 62, "63": 63, "64": 64, "65": 65,
                            "66": 66, "67": 67, "68": 68, "69": 69, "70": 70, "71": 71, "72": 72, "73": 73, "74": 74,
                            "75": 75, "76": 76, "77": 77, "78": 78, "79": 79, "80": 80, "81": 81, "82": 82, "83": 83,
                            "84": 84, "85": 85, "86": 86, "87": 87, "88": 88, "89": 89, "90": 90, "91": 91, "92": 92,
                            "93": 93, "94": 94, "95": 95, "96": 96, "97": 97, "996": 98, "997": 99, "998": 100,
                            "999": 101}
        # unknown codes (not mentioned in the code overview): none
        # 1. Current occupation

        # setup desired output path
        cocc_output_path = targetdir_codes + "/current_occupation_codes.csv"
        # generate variable names according to the ones mentioned in the respective dataset coding report
        cocc_select_vars = ["ID", "COCode"]
        # transform to binarized code dataframe and save
        cocc_df = transformer_function(path_to_df=occupation_codes,
                                       list_of_vars_selection=cocc_select_vars,
                                       mappings_dictionary=occupations_dict,
                                       num_code_vars=101,
                                       path_to_output=cocc_output_path)
        # match with verbatims and save
        vcocc_output_path = targetdir + "/current_occupation_verbatim_code.csv"
        current_occupation = verbatim_code_merger(verbatim_df=sheet_df_map['OccNow'],
                                                  binarized_codes_df=cocc_df,
                                                  output_path=vcocc_output_path)
        # 2. Past occupation

        # setup desired output path
        pocc_output_path = targetdir_codes + "/past_occupation_codes.csv"
        # generate variable names according to the ones mentioned in the respective dataset coding report
        pocc_select_vars = ("ID", "POCode")
        # transform to binarized code dataframe and save
        pocc_df = transformer_function(path_to_df=occupation_codes,
                                       list_of_vars_selection=pocc_select_vars,
                                       mappings_dictionary=occupations_dict,
                                       num_code_vars=101,
                                       path_to_output=pocc_output_path)
        # match with verbatims and save
        vpocc_output_path = targetdir + "/past_occupation_verbatim_code.csv"
        past_occupation = verbatim_code_merger(verbatim_df=sheet_df_map['OccPast'],
                                               binarized_codes_df=pocc_df,
                                               output_path=vpocc_output_path)
        # 3. Match the codes with the verbatims and stack the verbatims and codes for both into a dataset

        pieces = (past_occupation, current_occupation)  # order in verbatims excel-file
        occ_df_final = pd.concat(pieces, ignore_index=True)
        return_df_map['Dataset_opt_1_occupation'] = occ_df_final  # save to df map (additional dataset with new name)
        # sanity check: is sum of individual df row numbers equal final df row number?
        rows_sum = current_occupation.shape[0] + past_occupation.shape[0]
        print("(Optional dataset 1: occupation) Sanity check on row sum: ",
              rows_sum == occ_df_final.shape[0],
              "(Sum: ",
              occ_df_final.shape[0],
              ")")
        output_path = targetdir + "/occupation_complete_verbatim_codes.csv"
        occ_df_final.to_csv(output_path, index=False, header=True, encoding='utf-8')
        print("Finished Occupation (past/present): Occupation questions")

        # B. Industry codes and verbatims (past/present)
        # get input data filename
        industry_codes = rawdata_dir + "anes_timeseries_2008_occupation_industry/" + \
                         "Occupation and Industry codes.csv"

        # we have to go through each dataset individually, isolate the codings, transform them into a binary 1 x 101
        # vector (101 because we have 101 unique codes used in the dataset, see word coument on codes).
        # Finally, we can then merge them into one big dataset where all verbatims by question are stacked
        # under each other

        # setup the code to position mapping dict here:
        industry_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11,
                         "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20,
                         "21": 21, "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29,
                         "30": 30, "31": 31, "32": 32, "33": 33, "34": 34, "35": 35, "36": 36, "37": 37, "38": 38,
                         "39": 39, "40": 40, "41": 41, "42": 42, "43": 43, "44": 44, "45": 45, "46": 46, "47": 47,
                         "48": 48, "49": 49, "50": 50, "51": 51, "52": 52, "53": 53, "54": 54, "55": 55, "56": 56,
                         "57": 57, "58": 58, "59": 59, "60": 60, "61": 61, "62": 62, "63": 63, "64": 64, "65": 65,
                         "66": 66, "67": 67, "68": 68, "69": 69, "70": 70, "71": 71, "72": 72, "73": 73, "74": 74,
                         "75": 75, "76": 76, "77": 77, "78": 78, "79": 79, "80": 80, "81": 81, "82": 82, "83": 83,
                         "84": 84, "85": 85, "86": 86, "87": 87, "88": 88, "89": 89, "90": 90, "91": 91, "92": 92,
                         "93": 93, "94": 94, "95": 95, "96": 96, "97": 97, "98": 98, "99": 99, "996": 100, "997": 101,
                         "998": 102, "999": 103}

        # unknown codes (not mentioned in the code overview): none
        # 1. Current industry

        # setup desired output path
        cind_output_path = targetdir_codes + "/current_industry_codes.csv"
        # generate variable names according to the ones mentioned in the respective dataset coding report
        cind_select_vars = ["ID", "CICode"]

        # transform to binarized code dataframe and save
        cind_df = transformer_function(path_to_df=industry_codes,
                                       list_of_vars_selection=cind_select_vars,
                                       mappings_dictionary=industry_dict,
                                       num_code_vars=103,
                                       path_to_output=cind_output_path)
        # match with verbatims and save
        vcind_output_path = targetdir + "/current_industry_verbatim_code.csv"
        current_industry = verbatim_code_merger(verbatim_df=sheet_df_map['IndNow'],
                                                binarized_codes_df=cind_df,
                                                output_path=vcind_output_path)
        # 2. Past industry

        # setup desired output path
        pind_output_path = targetdir_codes + "/past_industry_codes.csv"
        # generate variable names according to the ones mentioned in the respective dataset coding report
        pind_select_vars = ["ID", "PICode"]

        # transform to binarized code dataframe and save
        pind_df = transformer_function(path_to_df=industry_codes,
                                       list_of_vars_selection=pind_select_vars,
                                       mappings_dictionary=industry_dict,
                                       num_code_vars=103,
                                       path_to_output=pind_output_path)
        # match with verbatims and save
        vpind_output_path = targetdir + "/past_industry_verbatim_code.csv"
        past_industry = verbatim_code_merger(verbatim_df=sheet_df_map['IndPast'],
                                             binarized_codes_df=pind_df,
                                             output_path=vpind_output_path)
        # 3. Match the codes with the verbatims and stack the verbatims and codes for both into a dataset

        pieces = (past_industry, current_industry)
        indu_df_final = pd.concat(pieces, ignore_index=True)
        return_df_map['Dataset_opt_2_industry'] = indu_df_final  # save to df map (additional dataset with new name)
        # sanity check: is sum of individual df row numbers equal final df row number?
        rows_sum = current_industry.shape[0] + past_industry.shape[0]
        print("Sanity check on row sum: ",
              rows_sum == indu_df_final.shape[0],
              "(Sum: ",
              indu_df_final.shape[0],
              ")")
        output_path = targetdir + "/industry_complete_verbatim_codes.csv"
        indu_df_final.to_csv(output_path, index=False, header=True, encoding='utf-8')
        print("Finished Industry (past/present): Industry questions")

    ########################################### RETURN THE MERGED VERBATIM/CODES DATASETS ##############################
    print("Done loading and preparing the datasets!")
    return return_df_map

if __name__ == "__main__":  #
    # notify user that this is just a data-preparation module
    print("This module only provides data preparation functions, "
          "please import into another script for working with the ANES 2008 (Open Ended Coding Project)- data!")