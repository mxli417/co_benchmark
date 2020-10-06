# Repository for the Paper: "A new benchmark for NLP in Social Sciences: Evaluating the usefulness of pre-trained language models for classifying open-ended survey responses"

This is the repository complementing our Paper submitted to the ICAART conference. 

We provide our used data in the folders "clean_data/" and " traintest_data/". The first contains the cleaned
and matched verbatim-codes data for all of our 10 thematic data sets. The second contains
the actually used train/test splits which have been split using a method developed by [1] / [2]
and implemented by [3]. We also provide the used python-code under "src/". 

The original, raw data come from the "Open Ended Coding Project" which accompanied the 2008 
ANES survey conducted by [4]. All credit for the initial collection of the open-ended data 
goes to ANES and the principal investigators: Jon Krosnick, Matt Berent and Arthur Lupia.
The original raw data can be obtained via the ANES-website for the ["Open Ended Coding Project"](https://electionstudies.org/2008-open-ended-coding-project/)
and the ["2008 Time Series Study"](https://electionstudies.org/data-center/2008-time-series-study/).

The creation of the cleaned data sets is described in our paper. For a quick overview
on the used method, consider the following: 

In order to generate data sets usable for multi-label learning from the files distributed 
by the Open Ended Coding Project, we exploit the notion of representing the codes, which 
have been assigned to each textual observation, by a binary vector. 

Additionally, to map the numeric codes to binary label vector elements one-to-one, 
we sourced the total size of each code set from the codes-documents enclosed with each data set. 
Using this information, we defined the length of the binary mapping vectors to be identical to 
the cardinality of the code sets. To generate k-hot encoded label vectors for each response contained 
in the data sets, we designed a mapping dictionary for each code set defining which code from the current
set belongs to which element in the binary vector generated for a particular response. 

To finally obtain the binary label vectors from the set of numeric codes associated to each observation, 
we designed a custom function which can be fed a mapping dictionary and the raw data row-by-row. 
The function then returns the binary label vectors of length q for each observation, where each vector 
element is 1 if the code mapped to this element was assigned to the response and 0 otherwise. All the 
data sets are transformed by using this function. For the latter application of machine learning methods
we split the data into train and test set (90/10) using an iterative stratification method for balancing
the label distributions, developed by [1]/[2] and implemented by [3] in scikit-multilearn.

## Literature:
 
1. Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data. Machine Learning and Knowledge Discovery in Databases, 145-158. http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf

2. Piotr Szymański, Tomasz Kajdanowicz (2017). Proceedings of the First International Workshop on Learning with Imbalanced Domains: Theory and Applications, PMLR 74:22-35, 2017. http://proceedings.mlr.press/v74/szyma%C5%84ski17a.html

3. Piotr Szymański, Tomasz Kajdanowicz (2017). A scikit-based Python environment for performing multi-label classification. ArXiv e-prints. https://arxiv.org/abs/1702.01460

4. The American National Election Studies (ANES). ANES 2008 Time Series Study. Ann Arbor, MI: Inter-university Consortium for Political and Social Research [distributor], 2015-11-10. https://doi.org/10.3886/ICPSR25383.v3


## Preliminary Leaderboard: 

Authors || Score | Dataset-ID | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
--------||------------|------------|---|---|---|---|---|---|---|---|---|---|
Card/Smith (2015)|| F1 (sample-based) |0.55|0.67|0.71|0.71|0.81|0.86|0.94|0.96|0.93|0.96|

