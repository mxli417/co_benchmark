"""
This is the optional code block which can convert the created datasets from the preprocessing stage into
sparse matrix (scipy lilmatrix sparse) - representations. This format is standard in scitki-multilearn
and the created datasets can be used with it. However, this is not an integral or necessary part of the
master thesis and hence is included into the OPTIONAL-block.
"""

import os
import pandas as pd
import numpy as np
import scipy
from collections import defaultdict
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
from skmultilearn.dataset import load_dataset_dump, save_dataset_dump


# 0. define helper functions

# convert the dataset to the skmultilearn format
# after vectorizing their verbatim contents

def data_converter(dataset, output_filename, label_names, format="skmulti"):
    """This only works for the special dataset structure of the project,
        having a caseID and verbatim variable & many code variables thereafter.
        Same structure in each dataset is assumed as given. The verbatims
        are vectorized and included in the final dataset as binary features.
    :param dataset: a pandas dataframe
    :param output_filename: a path to where the results will be save
    :param label_names: optional label names for each label present in the dataframe
    :param format: the desired output format (currently supports only skmulti-format)
    :return: boolean, True if conversion and storing on HDD is successful
    """
    # init preliminaries
    keycheck_1, keycheck_2 = False, False

    # check if dataset is given
    if dataset is None:
        print("[SK-MULTI-CONVERTER] No dataset given, aborting!")
        return

    if output_filename is None:
        print("[SK-MULTI-CONVERTER] No output filename given, aborting!")
        return
    else:
        # check the file ending in the output-filename
        file_ext = os.path.splitext(output_filename)[1]
        if not file_ext == '.bz2':
            print("[SK-MULTI-CONVERTER] False file extension in output_filename, aborting!")
            return

    if label_names is None:
        print("[SK-MULTI-CONVERTER] Optional label names not given, replacing names with dataframe-labels!")
        label_names = dataset.columns.values.tolist()  # use the header of the dataframe as label names

    # currently only one option avalaible: "skmulti"
    # (out of three pssible ones: "skmulti", "arff", "both" for possible output formats)
    if format == 'skmulti':
        # split the dataset up into two parts: X and y
        X = dataset.iloc[:, :2]  # vars: caseID and verbatim
        y = dataset.iloc[:, 2:]  # vars: all labels
        # vectorize the contents of the verbatim column in the X dataframe
        get_texts = X.verbatim.values.tolist()  # get raw strings list
        verbatim_words_list = []
        # split into protowords for each string in the list
        for row_index in range(len(get_texts)):
            raw_verbatim = get_texts[row_index]
            verbatim_wordlist = [word for word in raw_verbatim.split()]
            verbatim_words_list.append(verbatim_wordlist)
        # create a dictionary containing a mapping from each word in the corpus to an integer
        verbatim_dictionary = corpora.Dictionary(verbatim_words_list)
        # now init a vectorizer with the dictionary for the mappings
        verbatim_vectorizer = CountVectorizer(encoding='UTF-8',
                                              stop_words=None,  # override standard stopwords
                                              vocabulary=verbatim_dictionary.token2id)  # init count vectorizer from skl
        # vectorize the list of words in each verbatim list
        vectorized_verbatims = verbatim_vectorizer.transform(get_texts)
        verbatim_feature_names = verbatim_vectorizer.get_feature_names()
        print("[SK-MULTI-CONVERTER] Extracted feature names!")
        # compare the verbatim dictionary keys to the verbatim feature names,
        # if all are equal, we have a good vectorization at hand!
        orig_keylist = [kkey for kkey in verbatim_dictionary.token2id]
        new_keylist = verbatim_feature_names

        # check if the lists are equally long
        if len(new_keylist) == len(orig_keylist):
            keycheck_1 = True
            equality_counter = 0
            for index in range(len(new_keylist)):
                if orig_keylist[index] == new_keylist[index]:
                    equality_counter += 1
            # check if the equality counter equals the length
            # of the lists
            if equality_counter == len(new_keylist):
                keycheck_2 = True

        # evaluate if the mapping dictionary keys have been checked and marked as identical
        if keycheck_1 and keycheck_2:
            # inform the user
            print("[SK-MULTI-CONVERTER] Extracted keys are equal to the used keys!")
            # combine the caseID matrix with the sparse verbatim representations matrix
            get_np_column = np.array(X['caseID'])[:, None]  # re-shape the dataframe column to a numpy column
            combined_matrix = scipy.sparse.hstack((get_np_column, vectorized_verbatims))  # prepend the caseID variable
            combined_matrix = combined_matrix.tocsr()  # transform back to csr matrix
            # convert the X- and y-matrix to sparse lil matrix
            final_X = scipy.sparse.lil_matrix(combined_matrix)  # vectorized_verbatims
            final_y = scipy.sparse.lil_matrix(y)
            # construct the feature information for both X and y
            feature_tuples = [(element, [0, 1]) for element in verbatim_feature_names]
            X_feature_names = ['caseID', feature_tuples]
            y_feature_names = [(element, [0, 1]) for element in y.columns.values.tolist()]
            # save the dataset in new format - use the skmulti function for this
            save_dataset_dump(final_X, final_y, X_feature_names, y_feature_names, output_filename)
            # cross check if the file exists
            if not os.path.isfile(output_filename):
                print("File {} does not exist".format(output_filename))
                print("[SK-MULTI-CONVERTER] File could not be saved, aborting!")
            else:
                # give the user feedback
                print("[SK-MULTI-CONVERTER] Saved the given dataset in "
                      "skmultilearn-format at: \n {}".format(output_filename))
                return True
        else:
            print("[SK-MULTI-CONVERTER] Fatal error, breaking! (no identity in mapping keys!)")
            return False


# also write a function which loads the vectorized text data (with y features)
# loads a back-mapping dictionary from the feature names and then re-constructs
# the original texts - then finally run a unit test which loads all datasets and
# checks the vectorization with the re-mapping function. If successful, dump converted
# datasets (below):

def data_deconverter(dataset_filename, orig_dataset):
    """A function that loads a previously converted dataframe from the specified path.
    The prints the loaded data and the original data side-by-side.

    :param dataset_filename: path to where the file has been stored
    :param orig_dataset: original dataset as pandas dataframe for comparison
    :return: boolean, True if data loadup and comparison is successful
    """
    if os.path.isfile(dataset_filename):
        print("[DECONVERTER] Loading data from filename: {}".format(dataset_filename))
        X, y, X_feature_names, y_feature_names = load_dataset_dump(dataset_filename)
        # get the relevant parts of the X matrix for inverse vectorization
        Dense_X = X.todense()
        Subset_token_features = Dense_X[:, 1:]
        # get all relevant word features
        X_feature_names = X_feature_names[1:]
        X_feature_names = X_feature_names[0]
        # enumerate all the features, extract the names
        mapping_dict = defaultdict()
        # extract the mapping features and build a mapping dict
        mapping_integer = 0
        for element in X_feature_names:
            raw_element = element[0]  # only extract the word part
            # create a dictionary entry with a mapping integer
            mapping_dict[raw_element] = mapping_integer
            # increment
            mapping_integer += 1
        # use the re-created mapping dict to generate word list per verbatim
        # inverse transform the X matrix to terms
        re_vectorizer = CountVectorizer(vocabulary=mapping_dict)  # init the vectorizer
        re_vectorizer.fit(mapping_dict)  # fit on the extracted vocabulary
        deconverted_data = re_vectorizer.inverse_transform(Subset_token_features)
        verbatim_list = []
        # build a list of lists from this
        for element in deconverted_data:
            new_element = element.tolist()
            verbatim_list.append(new_element)
        # print side by side
        if len(verbatim_list) == orig_dataset.shape[0]:
            indexer = 0
            for indexer in range(len(verbatim_list)):
                print(verbatim_list[indexer])
                print(orig_dataset.iloc[indexer, 1])
        return True
    else:
        print("[DECONVERTER] No file found at: {}".format(dataset_filename))
        return False


################################################ MAIN ##################################################################

# Please use this section if you cannot or do not want to use the master script
"""###################### 1. setup preliminaries: paths, variables, etc. ###############################################
data_source_dir_normal = "C:/Users/mxmd/PycharmProjects/master_thesis/data/"
output_path = "C:/Users/mxmd/PycharmProjects/master_thesis/data/sparse_data" """


# 2. run conversion
def main_converter(data_source_dir_normal, output_path):
    # include the functionality from before here
    print("Starting data converter for all datasets (1-10)!")
    # generate all load-up data set paths
    pathlist = []
    for i in range(1, 11):
        gen_name = data_source_dir_normal + 'dataset{}.csv'.format(i)
        pathlist.append(gen_name)
    # generate the output paths
    out_list = []
    for i in range(1, 11):
        out_name = output_path + 'skmulti_ds{}.bz2'.format(i)
        out_list.append(out_name)
    # load and convert the datasets
    success_list = []
    for index, element in enumerate(pathlist):
        # get data
        df_load = pd.read_csv(element, encoding='utf-8')
        # get output path
        fname_out = out_list[index]
        # convert and deconvert (without additional label descriptions)
        if data_converter(df_load, label_names=None, output_filename=fname_out, format='skmulti'):
            fpath_split = os.path.split(element)[1]
            print("Data conversion success for data set {}!".format(fpath_split))
            success_list.append(1)
        else:
            success_list.append(0)
        # deconvert the dataset and print out the verbatims
        # data_deconverter(fname_out, orig_dataset=df_load) # excluded for convenience
    # check if everything was successful (all conversions successful)
    if sum(success_list) == 10:
        return True
    else:
        return False

if __name__ == "__main__":  #
    # notify user that this is just a data-preparation module
    print("This module only provides data converter functions, "
          "please import into another script for working with the ANES 2008 (Open Ended Coding Project)-data!")
