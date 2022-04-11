# -----------------------------------------------------------------------------
#             FUNCTION FOR PREPROCESSING DATA FOR ML PIPELINE
# -----------------------------------------------------------------------------
# (C) Nicolas Rost, 2022

import pandas as pd
import numpy as np
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
import pyper as pr
from missings import plot_removed_missings


def preprocess_data(data,
                    target,
                    na_cutoff_samples = 0.75,
                    na_cutoff_features = 0.5,
                    min_count_binary = 0.05,
                    include_bio = False):
    '''
    This function contains several preprocessing steps for the MARS ML data.

    Please specify 1) the dataframe name
                   2) the target you want to predict
                   3) the maximum number of missings allowed per sample
                   4) the maximum number of missings allowed per feature
                   5) the minimum count of the minor category of binary features (0/1) in percent
                   6) if biological variables should be included or not
    '''

    # 1. + 2. SPLIT INTO FEATURES AND TARGET
    outcomes = ['HMD17me_06_changeperc',
                 'HMD17me_06',
                 'HMD17me_card_changeperc',
                 'HMD17me_card_06',
                 'S3_categ_diff_06',
                 'KRS17me_6W',
                 'KRM17me_6W']
    outcomes.remove(target) # remove outcome that we want to keep
    data = data.drop(data.loc[:, outcomes], axis = 1)

    obj_cols = data.columns[data.dtypes.eq('object')]  # for some reason, some columns are of type object --> make them numeric
    data[obj_cols] = data[obj_cols].apply(pd.to_numeric, errors = 'coerce')

    outcome = data[[target]]
    features = data.drop(target, axis = 1)

    if target == 'KRS17me_6W':
        outcome[[target]] = -outcome[[target]] # needs to be recoded

    # 3. SAMPLE SELECTION
    temp = deepcopy(data)
    n_before = temp.shape[0] # number of samples before
    plot_removed_missings(temp) # plot remaining samples per amount to select thresh
    n_vars = temp.shape[1] # number of variables
    temp = temp[temp.isnull().sum(axis = 1) <= (na_cutoff_samples * n_vars)]
    n_after = temp.shape[0] # number of samples afterwards
    n_diff = n_before - n_after # difference
    print(f'\nRemoved {n_diff} samples because of >= {na_cutoff_samples * 100} % missing values.')
    # split into outcome and features again
    outcome = temp[[target]]
    features = temp.drop([target], axis = 1)

    # 4. REMOVE FEATURES WITH TOO MANY MISSINGS
    ncol_before = features.shape[1] # number of features before
    features = features.loc[:, features.isnull().mean() < na_cutoff_features]
    ncol_after = features.shape[1] # number of features afterwards
    ncol_diff = ncol_before - ncol_after # difference
    print(f'\nDeleted {ncol_diff} features because of >= {na_cutoff_features * 100} % missing values.')

    # 5. REMOVE FEATURES WITH LOW VARIANCE
    ncol_before = features.shape[1] # number of features before
    for f in features.columns:
        if (len(features[f].value_counts()) == 1) or ((len(features[f].value_counts()) == 2) and (features[f].value_counts(normalize = True).min() < min_count_binary)):
            features.drop(f, inplace = True, axis = 1)
    ncol_after = features.shape[1] # number of features afterwards
    ncol_diff = ncol_before - ncol_after # difference
    print(f'\nDeleted {ncol_diff} one-hot-encoded features because too unilaterally distributed (> 95:5)')

    # 6. REMOVE BIOLOGICAL VARIABLES
    if include_bio == False:
        ncol_before = features.shape[1] # number of features before
        features = features.drop(features[features.columns[pd.Series(features.columns).str.startswith('PRS')]], axis = 1, errors = 'ignore')
        vars_bio = ['size', 'bmi_00', 'HF_00', 'RRsys_00', 'RRdia_00', 'EKG_00', 'Gew_00', 'WAIST_00', 'HIPP_00',
                    'BZ_00', 'Harns_00', 'trigl_00', 'Chol_00', 'TSH_00', 'fT3_00', 'fT4_00', 'Cortisol', 'hsCRP_log', 'IL6_log']
        features = features.drop(vars_bio, axis = 1, errors = 'ignore')
        ncol_after = features.shape[1] # number of features afterwards
        ncol_diff = ncol_before - ncol_after # difference
        print(f'\nDeleted {ncol_diff} biological variables (e.g., PRS and serum data).')

    return features, outcome
