# --------------------------------------------------------------------------
#                     CLINICAL DATA DESCRIPTIVE STATS
# --------------------------------------------------------------------------
# (C) Nicolas Rost, 2022


# libraries
import pandas as pd


def summarize_clinical_info(data, save = False):
    """
    Function that shows descriptive summary of clinical features.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the clinical features and all their data.
        Clinical training features. Simulation training data will be created in here again.
    save : BOOL, optional
        Should the summary dataframe be saved? The default is False.

    Returns
    -------
    Pandas dataframe with descriptive summary information.

    """

    # descriptives data frame
    d = data.describe()
    # add number and % of missings
    miss_n    = []
    miss_perc = []
    rows = data.shape[0]
    for col in d:
        d.loc['min',col] = '{:.2f}'.format(round(d.loc['min',col], 2)) # round already to 2 digits
        d.loc['max',col] = '{:.2f}'.format(round(d.loc['max',col], 2)) # round already to 2 digits
        count = d.loc['count',col]
        miss_n.append(rows - count)
        miss_perc.append('{:.2f}'.format(round((rows - count) / rows * 100, 2))) # round already to 2 digits
    d.loc[len(d)] = miss_n
    d.loc[len(d)] = miss_perc

    # reformat
    desc = pd.DataFrame({'database_name':d.columns,
                         'missings':d.loc[8,:].astype(int).astype(str) + ' (' + d.loc[9,:] + ')',
                         'range':d.loc['min',:] + '-' + d.loc['max',:]})
    desc.reset_index(drop = True, inplace = True)
    desc['missings'] = desc['missings'].replace({'0 (0.00)':'0 (0)'})
    desc['range'] = desc['range'].replace(['\.00'], '', regex = True)

    # load feature names for merging
    names = pd.read_excel('MARS_feature_names.xlsx')
    description = names[['database_name', 'feature_name', 'description']]
    comments    = names[['database_name', 'comments']]
    # merge everything
    df = pd.merge(description, desc, on = 'database_name')
    df = pd.merge(df, comments, on = 'database_name')
    df = df.drop('database_name', axis = 1)

    if save:
        df.to_excel('clinical_features_descriptives.xlsx', index = False)

    return df
