# -----------------------------------------------------------------------------
#        FUNCTIONS FOR INFORMATION ON MISSING VALUES OVER SAMPLES
# -----------------------------------------------------------------------------
# (C) Nicolas Rost, 2022

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyper as pr


# ---------- COMPARE AMOUNT OF MISSINGS PER VARIABLE BETWEEN GROUPS -----------
def compare_missings(data, groups, variables = None, path_to_R = "C:/PROGRA~1/R/R-4.1.1/bin/x64/R"):
    '''
    This function uses Fisher's exact tests to compare the amount
    of missing values over single or multiple variables per group.

    Indicated your dataframe, your "group" variable and variables you want
    to test.

    If variables is set to None, the total number of missings over all other columns
    in the dataframe are used.
    '''

    results = dict()

    if variables is not None:

        for v in variables:

            if data[v].isna().sum() == 0:
                continue # skip variables without any missings

            ct = pd.crosstab(data[groups], np.isnan(data[v]))

            r = pr.R(RCMD = path_to_R)
            r.assign("ct", ct)
            dif = r.get("fisher.test(ct, workspace = 2e6)")

            if dif['p.value'] < 0.05:
                print("Significant group difference detected for: {}".format(v))

            results[v] = dict()
            results[v]['missings'] = ct
            results[v]['fisher_test'] = dif

    else:

        n_all = data[groups].value_counts() * (data.shape[1] - 1) # overall number of entries per group
        miss_all = data.drop(groups, axis = 1).isna().groupby(data[groups]).sum().sum(axis = 1) # number of overall missings per group

        ct = pd.DataFrame({'False': n_all - miss_all, 'True': miss_all})

        r = pr.R(RCMD = path_to_R)
        r.assign("ct", ct)
        dif = r.get("fisher.test(ct, workspace = 2e6)")

        if dif['p.value'] < 0.05:
            print("Significant group difference detected: p = {}".format(dif['p.value']))

        results['missings'] = ct
        results['fisher_test'] = dif

    return results


# --------- PLOT SAMPLE SIZE IF YOU REMOVE SAMPLES BECAUSE OF MISSINGS --------
# calculation and plotting
def plot_removed_missings(data):
    '''
    This function plots how many samples remain
    if you remove samples by their amount of missing values.

    Currently, steps of 5% are plotted.

    Just input a pandas dataframe of your choice.
    '''

    # number of variables
    n_samp = data.shape[0]
    n_vars = data.shape[1]

    # define percentage steps from 100 to 0 by 5%
    steps = np.arange(100, -1, -5) / 100

    # create list and save remaining samples per step
    samples = []
    for s in steps:
        data_temp = data[data.isnull().sum(axis = 1) <= (s * n_vars)]
        sample = data_temp.shape[0]
        samples.append(sample)

    # create dataframe for plotting
    df = pd.DataFrame({'step':steps*100, 'samples':samples})

    # creat plot
    sns.set(style = 'whitegrid', font_scale = 1.25)
    ax = sns.lineplot(x = 'step', y = 'samples', data = df, marker = 'o')
    ax.set_title('overall N: {}'.format(n_samp))
    ax.set(xlabel = 'maximum % of NA allowed', ylabel = 'remaining samples')
    ax.set_xticks(np.arange(0, 101, 10))
    #plt.savefig('sample_reduction_missings.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
