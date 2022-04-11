# --------------------------------------------------------------------------
#        PLOT NUMBER OF FEATURES SELECTED FROM ML MODELS
# --------------------------------------------------------------------------
# (C) Nicolas Rost, 2022

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from plotnine import *
from itertools import chain
from sklearn.model_selection import train_test_split
from utils.data_simulation import create_simulated_data
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef



def plot_feature_numbers(Xtrain, ytrain, Xtest, ytest,
                         save = False):
    """
    Function to plot the number of features selected by the ML pipeline models.
    Does not include permutations.

    Parameters
    ----------
    Xtrain : np.array
        Clinical training features. Simulation training data will be created in here again.
    ytrain : np.array
        Clinical training target. Simulation training data will be created in here again.
    Xtest : np.array
        Clinical test features. Simulation test data will be created in here again.
    ytest : np.array
        Clinical test target. Simulation test data will be created in here again.
    save : BOOL, optional
        Should the plot be saved. The default is False.

    Returns
    -------
    Pandas dataframe for plotting.

    """

    # create simulated data
    X, y = create_simulated_data()
    indices = range(X.shape[0])
    Xtrain_sim, Xtest_sim, ytrain_sim, ytest_sim, train_idx, test_idx = train_test_split(X, y, indices, test_size = 0.2, random_state = 1897)

    # load all results except for permutations
    files = [f for f in Path.cwd().glob("results/results_pipeline_*") if not "permutations" in f.name]
    results = {}
    feats = {}
    for f in files:
        temp = open(f, 'rb')
        name = str(f.name).replace("results_pipeline_", "")
        results[name] = pickle.load(temp)
        if 'no_rfe' in name:
            feats[name] = pickle.load(temp)
        else: # for RFE, we also need to remove those with 0 weights
            m_rfe = results[name].best_estimator_.steps[1][1].estimator_
            if 'clin' in name:
                Xtrain_red = Xtrain[:, results[name].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
                Xtest_red = Xtest[:, results[name].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
                m_rfe.fit(Xtrain_red, ytrain)
            if 'sim' in name:
                Xtrain_sim_red = Xtrain_sim[:, results[name].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
                Xtest_sim_red = Xtest_sim[:, results[name].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
                m_rfe.fit(Xtrain_sim_red, ytrain_sim)
            if any(x in name for x in ['sgdc', 'svc']):
                weights = results[name].best_estimator_.steps[1][1].estimator_.steps[1][1].coef_
                feats[name] = weights[weights != 0]
            if 'rfcl' in name:
                weights = results[name].best_estimator_.steps[1][1].estimator_.steps[1][1].feature_importances_
                feats[name] = weights[weights != 0]
        temp.close()

    # extract number of features and put them into dataframe
    # order of the first 3 columns has to be alphabetically because that's how it's ordered by Pathlib above!
    df = pd.DataFrame({
        'data': [x for x in ['Clinical data', 'Simulated data'] for i in range(6)],
        'clf': ['RF', 'RF', 'LR', 'LR', 'SVC', 'SVC'] * 2,
        'model': ['no RFE', 'RFE'] * 6,
        'n_feat': [feats[x].shape[0] for x in feats]
        })

    # plot data as barplots
    p = (
    ggplot(df, aes(x = 'model', y = 'n_feat', fill = 'model')) +
    geom_bar(stat = 'identity', alpha = 0.75) +
    geom_text(aes(label = 'n_feat', y= 'n_feat'))+
    scale_x_discrete(name = "") +
    scale_y_continuous(name = "Number of features", limits = (0, 135), breaks = np.arange(0, 121, 20)) +
    scale_fill_manual(values = ['#1C4774', '#4075AD']) +
    coord_flip() +
    facet_grid('data~clf') +
    theme_bw() +
    theme(axis_title_x = element_text(size = 12),
          axis_text_x = element_text(size = 10, angle = 45),
          axis_title_y = element_text(size = 12),
          axis_text_y = element_text(size = 10),
          strip_text_x = element_text(size = 10),
          strip_text_y = element_text(size = 10),
          legend_position = 'none')
    )
    if save:
        ggsave(p, filename = 'plots\\barplot_feature_numbers.png', height = 8, width = 15, units = 'cm', dpi = 900)

    return df


def get_feature_importances(clin_features,
                            Xtrain, ytrain, Xtest, ytest, n_repeats,
                            save = False):
    """
    Function extract the features importances for all features over the models.
    Does not include the permutations.

    Parameters
    ----------
    clin_features: pd.DataFrame
        Clinical features, needed for names. Feature names of the simulated data will be recreated in here.
    Xtrain : np.array
        Clinical training features. Simulation training data will be created in here again.
    ytrain : np.array
        Clinical training target. Simulation training data will be created in here again.
    Xtest : np.array
        Clinical test features. Simulation test data will be created in here again.
    ytest : np.array
        Clinical test target. Simulation test data will be created in here again.
    n_repeats: INT
        Number of permutations.
    save : BOOL, optional
        Should the results be pickled? The default is False.

    Returns
    -------
    Dictionary and dataframes with feature names and permutation importances for all models.

    """

    # load all results except for permutations
    files = [f for f in Path.cwd().glob("results/results_pipeline_*") if not "permutations" in f.name]
    results = {}
    feats = {}
    for f in files:
        temp = open(f, 'rb')
        name = str(f.name).replace("results_pipeline_", "")
        results[name] = pickle.load(temp)
        feats[name] = pickle.load(temp)
        temp.close()

    # recreate simulation data
    X_sim, y_sim = create_simulated_data()
    # additionally, we need to create arbitrary feature names for importances below
    sim_features = pd.DataFrame(X_sim, columns = [f"feature_{var + 1:03d}" for var in range(X_sim.shape[1])])
    Xtrain_sim, Xtest_sim, ytrain_sim, ytest_sim = train_test_split(X_sim, y_sim, test_size = 0.2, random_state = 1897)

    # calculate permutation importance for all features and all models
    importances = {}
    clin_table = pd.DataFrame({'feature':clin_features.columns.values}) # table format with string cells (M, SD)
    sim_table = pd.DataFrame({'feature':sim_features.columns.values}) # table format with string cells (M, SD)
    clin_imps = np.empty((0, n_repeats)) # will be turned into a DataFrame with all importances from all permutations
    sim_imps = np.empty((0, n_repeats)) # will be turned into a DataFrame with all importances from all permutations
    clin_model_list = [] # list for saving model names
    sim_model_list = [] # list for saving model names
    clin_features_list = [] # list for saving features
    sim_features_list = [] # list for saving features
    # loop over models, calculated importances for the respective 'clin' or 'sim' data with or without RFE, and save results
    for model in results:
        if 'no_rfe' in model:
            m = results[model].best_estimator_
            if 'clin' in model:
                m.fit(Xtrain, ytrain)
                importances[model] = permutation_importance(m,
                                                            Xtest,
                                                            ytest,
                                                            n_repeats = n_repeats,
                                                            random_state = 1897,
                                                            n_jobs = -1)
                importances[model]['features'] = list(clin_features.columns.values)
                means = importances[model]['importances_mean'].round(3).astype(str).tolist()
                stds = importances[model]['importances_std'].round(3).astype(str).tolist()
                mean_std = ['{} ({})'.format(a, b) for a, b in zip(means, stds)]
                temp_df = pd.DataFrame({'feature':importances[model]['features'], model:mean_std})
                clin_table = pd.merge(clin_table, temp_df, on = 'feature', how = 'left')
                clin_imps = np.append(clin_imps, importances[model]['importances'], axis = 0)
                clin_model_list.extend([model] * len(importances[model]['features']))
                clin_features_list.extend(importances[model]['features'])
            if 'sim' in model:
                m.fit(Xtrain_sim, ytrain_sim)
                importances[model] = permutation_importance(m,
                                                            Xtest_sim,
                                                            ytest_sim,
                                                            n_repeats = n_repeats,
                                                            random_state = 1897,
                                                            n_jobs = -1)
                importances[model]['features'] = list(sim_features.columns.values)
                means = importances[model]['importances_mean'].round(3).astype(str).tolist()
                stds = importances[model]['importances_std'].round(3).astype(str).tolist()
                mean_std = ['{} ({})'.format(a, b) for a, b in zip(means, stds)]
                temp_df = pd.DataFrame({'feature':importances[model]['features'], model:mean_std})
                sim_table = pd.merge(sim_table, temp_df, on = 'feature', how = 'left')
                sim_imps = np.append(sim_imps, importances[model]['importances'], axis = 0)
                sim_model_list.extend([model] * len(importances[model]['features']))
                sim_features_list.extend(importances[model]['features'])
        else:
            m = results[model].best_estimator_.steps[1][1].estimator_
            if 'clin' in model:
                Xtrain_red = Xtrain[:, results[model].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
                Xtest_red = Xtest[:, results[model].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
                m.fit(Xtrain_red, ytrain)
                importances[model] = permutation_importance(m,
                                                            Xtest_red,
                                                            ytest,
                                                            n_repeats = n_repeats,
                                                            random_state = 1897,
                                                            n_jobs = -1)
                importances[model]['features'] = list(clin_features.columns.values[results[model].best_estimator_.steps[1][1].support_])
                means = importances[model]['importances_mean'].round(3).astype(str).tolist()
                stds = importances[model]['importances_std'].round(3).astype(str).tolist()
                mean_std = ['{} ({})'.format(a, b) for a, b in zip(means, stds)]
                temp_df = pd.DataFrame({'feature':importances[model]['features'], model:mean_std})
                clin_table = pd.merge(clin_table, temp_df, on = 'feature', how = 'left')
                clin_imps = np.append(clin_imps, importances[model]['importances'], axis = 0)
                clin_model_list.extend([model] * len(importances[model]['features']))
                clin_features_list.extend(importances[model]['features'])
            if 'sim' in model:
                Xtrain_sim_red = Xtrain_sim[:, results[model].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
                Xtest_sim_red = Xtest_sim[:, results[model].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
                m.fit(Xtrain_sim_red, ytrain_sim)
                importances[model] = permutation_importance(m,
                                                            Xtest_sim_red,
                                                            ytest_sim,
                                                            n_repeats = n_repeats,
                                                            random_state = 1897,
                                                            n_jobs = -1)
                importances[model]['features'] = list(sim_features.columns.values[results[model].best_estimator_.steps[1][1].support_])
                means = importances[model]['importances_mean'].round(3).astype(str).tolist()
                stds = importances[model]['importances_std'].round(3).astype(str).tolist()
                mean_std = ['{} ({})'.format(a, b) for a, b in zip(means, stds)]
                temp_df = pd.DataFrame({'feature':importances[model]['features'], model:mean_std})
                sim_table = pd.merge(sim_table, temp_df, on = 'feature', how = 'left')
                sim_imps = np.append(sim_imps, importances[model]['importances'], axis = 0)
                sim_model_list.extend([model] * len(importances[model]['features']))
                sim_features_list.extend(importances[model]['features'])

    # now rename features into more publishable acronyms
    names = pd.read_excel('MARS_feature_names.xlsx')
    translations = names[['database_name', 'feature_name']]
    translations = translations.rename(columns = {'database_name':'feature'})
    clin_table = pd.merge(translations, clin_table, on = 'feature', how = 'inner')
    clin_table = clin_table.drop('feature', axis = 1)

    # reformat importances, models, and features from single permutations into long format (for plotting)
    df_clin = pd.DataFrame(clin_imps)
    df_clin['model'] = clin_model_list
    df_clin['feature'] = clin_features_list
    df_clin['classifier'] = np.where(df_clin['model'].str.contains('sgdc'), 'LR', np.nan)
    df_clin['classifier'] = np.where(df_clin['model'].str.contains('rfcl'), 'RF', df_clin['classifier'])
    df_clin['classifier'] = np.where(df_clin['model'].str.contains('svc'), 'SVC', df_clin['classifier'])
    df_clin['model'] = np.where(df_clin['model'].str.contains('no_rfe'), 'no RFE', 'RFE')
    df_clin = pd.melt(df_clin, id_vars = ['classifier', 'model', 'feature'], var_name = 'permutation', value_name = 'importance')
    df_clin = pd.merge(translations, df_clin, on = 'feature', how = 'inner')
    df_clin = df_clin.drop('feature', axis = 1)
    df_sim = pd.DataFrame(sim_imps)
    df_sim['model'] = sim_model_list
    df_sim['feature'] = sim_features_list
    df_sim['classifier'] = np.where(df_sim['model'].str.contains('sgdc'), 'LR', np.nan)
    df_sim['classifier'] = np.where(df_sim['model'].str.contains('rfcl'), 'RF', df_sim['classifier'])
    df_sim['classifier'] = np.where(df_sim['model'].str.contains('svc'), 'SVC', df_sim['classifier'])
    df_sim['model'] = np.where(df_sim['model'].str.contains('no_rfe'), 'no RFE', 'RFE')
    df_sim = pd.melt(df_sim, id_vars = ['classifier', 'model', 'feature'], var_name = 'permutation', value_name = 'importance')
    df_sim = df_sim.rename(columns = {'feature':'feature_name'}) # make column name consistent with df_clin

    if save:
        f = open('results/permutation_importance', 'wb')
        pickle.dump(importances, f)
        pickle.dump(clin_table, f)
        pickle.dump(sim_table, f)
        pickle.dump(df_clin, f)
        pickle.dump(df_sim, f)
        f.close()

    return importances, clin_table, sim_table, df_clin, df_sim


def plot_features_by_performance(metric, Xtrain, ytrain, Xtest, ytest, save = False):
    """
    Function to plot the number of features selected by the ML pipeline models against the model performances.
    Does not include permutations.

    Parameters
    ----------
    metric = STR
        Enter 'BAC' or 'MCC'
    Xtrain : np.array
        Clinical features. Has to match 'data'.
    ytrain : np.array
        Clinical target. Has to match 'data'.
    Xtest : np.array
        Clinical features. Has to match 'data'.
    ytest : np.array
        Clinical target. Has to match 'data'.
    save : BOOL, optional
        Should the plot be saved. The default is False.

    Returns
    -------
    Pandas dataframe for plotting.

    """

    # load all results except for permutations
    files = [f for f in Path.cwd().glob('results/results_pipeline_*') if not 'permutations' in f.name]
    results = {}
    feats = {}
    for f in files:
        temp = open(f, 'rb')
        name = str(f.name).replace('results_pipeline_', '')
        results[name] = pickle.load(temp)
        if 'no_rfe' in name:
            feats[name] = pickle.load(temp)
        else: # for RFE, we also need to remove those with 0 weights
            if any(x in name for x in ['sgdc', 'svc']):
                weights = results[name].best_estimator_.steps[1][1].estimator_.steps[1][1].coef_
                feats[name] = weights[weights != 0]
            if 'rfcl' in name:
                weights = results[name].best_estimator_.steps[1][1].estimator_.steps[1][1].feature_importances_
                feats[name] = weights[weights != 0]
        temp.close()

    # recreate simulation data
    X_sim, y_sim = create_simulated_data()
    Xtrain_sim, Xtest_sim, ytrain_sim, ytest_sim = train_test_split(X_sim, y_sim, test_size = 0.2, random_state = 1897)

    # load performances and select metric
    f = open('results/model_performances', 'rb')
    model_performances = pickle.load(f)
    f.close()
    perf = {}
    for model in results:
        perf[model] = round(model_performances[metric][model], 3)

    # extract number of features and put them into dataframe
    # order of the first 3 columns has to be alphabetically because that's how it's ordered by Pathlib above!
    df = pd.DataFrame({
        'data': [x for x in ['Clinical data', 'Simulated data'] for i in range(6)],
        'Classifier': ['RF', 'RF', 'LR', 'LR', 'SVC', 'SVC'] * 2,
        'Method': ['no RFE', 'RFE'] * 6,
        'n_feat': [feats[x].shape[0] for x in feats],
        f'{metric}': [perf[x] for x in perf]
        })

    # plot as scatterplot
    if metric == 'BAC':
        ytitle = 'Balanced accuracy'
    if metric == 'MCC':
        ytitle = 'Matthews correlation coefficient'
    p = (
    ggplot(df, aes(x = 'n_feat', y = metric, color = 'Method', fill = 'Method', shape = 'Classifier')) +
    geom_point(size = 5, alpha = 0.75) +
    scale_x_continuous(name = 'Number of features', limits = (0, 130), breaks = np.arange(0, 121, 20)) +
    scale_y_continuous(name = ytitle, limits = (min(df[metric]) - 0.1, max(df[metric]) + 0.1)) +
    scale_color_manual(values = ['#1C4774', '#4075AD'] * 3) +
    scale_fill_manual(values = ['#1C4774', '#FFFFFF00'] * 3) +  # Add this
    facet_grid('~data') +
    theme_bw() +
    theme(axis_title_x = element_text(size = 12),
          axis_text_x = element_text(size = 10, angle = 45),
          axis_title_y = element_text(size = 12),
          axis_text_y = element_text(size = 10),
          strip_text_x = element_text(size = 10),
          legend_title = element_text(size = 12),
          legend_text = element_text(size = 10))
    )
    if save:
        ggsave(p, filename = f'plots\\scatterplot_features_{metric}.png', height = 8, width = 14, units = 'cm', dpi = 900)

    return df


def plot_feature_importances(importances, sort_by = 'alphabetically', save = False):
    """
    Function extract the features importances for all features over the models.
    Does not include the permutations.

    Parameters
    ----------
    importances: pd.DataFrame
        Dataframe with features and importances, output of get_feature_importances(). Needs the columns 'feature_name', 'importance', 'model', 'classifier'.
    sort_by: STR, optional
        String with either 'alphabetically' (default) or 'top_importance'. Indicates how the features in the plot should be sorted.
    save: bool, optional
        Should the plot be saved. The default is False.

    Returns
    -------
    Plot of feature importance.

    """

    allowed_sortings = ['alphabetically', 'top_importance']
    if sort_by not in allowed_sortings:
        raise ValueError('Unknown sorting entered. Please select one of the following: {}'.format(allowed_sortings))

    if sort_by == 'alphabetically':
        importances = importances.sort_values(by = 'feature_name', inplace = False, key = lambda col: col.str.lower()) # sort alphabetically
        h, w = 50, 12 # for plot
    if sort_by == 'top_importance':
        importances['mean_feature_importance'] = importances.groupby('feature_name')['importance'].transform(np.mean) # add overall mean importance (overall both methods and all classifiers)
        importances = importances[importances['feature_name'].isin(importances['feature_name'].mode())] # keep only features without zero importances
        importances = importances.sort_values(by = 'mean_feature_importance', ascending = False) # sort by mean importance
        importances = importances[importances['mean_feature_importance'] > 0] # keep only those with mean importances > 0
        h, w = 25, 12 # for plot

    p = (
    ggplot(importances, aes(x = 'feature_name', y = 'importance', color = 'model', shape = 'model')) +
    geom_hline(yintercept = 0, linetype = '--', color = 'grey', alpha = 0.75) +
    stat_summary(fun_data = 'mean_cl_normal', fun_args = {'mult':1}, geom = 'errorbar', alpha = 0.75, position = position_dodge(width = 0.8)) +
    stat_summary(fun_y = np.mean, geom = 'point', alpha = 0.75, position = position_dodge(width = 0.8)) +
    scale_x_discrete(name = '', limits = importances['feature_name'].unique().tolist()[::-1]) +
    scale_y_continuous(name = 'Permutation importance (M ± 95% CI)') +
    scale_color_manual(name = '', values = ['#1C4774', '#4075AD']) +
    coord_flip() +
    facet_grid('~classifier') +
    theme_bw() +
    theme(axis_title_x = element_text(size = 12),
          axis_text_x = element_text(size = 10, angle = 45),
          axis_title_y = element_text(size = 12),
          axis_text_y = element_text(size = 10),
          strip_text_x = element_text(size = 10),
          legend_title = element_blank())
    )
    if save:
        if 'feature_' in importances['feature_name'].unique()[0]: # check if simulated data or not
            ggsave(p, filename = f'plots\\features_importances_sim_{sort_by}.png', height = h, width = w, units = 'cm', dpi = 900)
        else:
            ggsave(p, filename = f'plots\\features_importances_clin_{sort_by}.png', height = h, width = w, units = 'cm', dpi = 900)
    return p


def plot_feature_importances_main(importances, save = False):
    """
    Almost same function as plot_feature_importances, just with clearer variable names for publication.
    Feature names are added manually here, so be careful: when feature importances change, names need to be adapted here.

    Parameters
    ----------
    importances: pd.DataFrame
        Dataframe with features and importances, output of get_feature_importances(). Needs the columns 'feature_name', 'importance', 'model', 'classifier'.
    save: bool, optional
        Should the plot be saved. The default is False.

    Returns
    -------
    Plot of feature importance.

    """

    importances['mean_feature_importance'] = importances.groupby('feature_name')['importance'].transform(np.mean) # add overall mean importance (overall both methods and all classifiers)
    importances = importances[importances['feature_name'].isin(importances['feature_name'].mode())] # keep only features without zero importances
    importances = importances.sort_values(by = 'mean_feature_importance', ascending = False) # sort by mean importance
    importances = importances[importances['mean_feature_importance'] > 0] # keep only those with mean importances > 0
    h, w = 25, 12 # for plot

    # rename features
    importances['feature_name'] = importances['feature_name'].replace({'n_hos':'number of prior hospitalizations',
                                                                       'fam_hist_F32':'family history of depressive episodes',
                                                                       'diet':'whether patient is put on a diet',
                                                                       'years_since_last_hos':'time since last hospitalization',
                                                                       'dur_episode':'duration of current episode',
                                                                       'HDRS_i12':'HDRS item 12: somatic symptoms gastro-intestinal',
                                                                       'n_grandp_nat_german':'number of native German speaking grandparents',
                                                                       'nonviolent_suicide_attempt':'nonviolent suicide attempts in medical history',
                                                                       'dysthymia':'preexisting dysthymia',
                                                                       'suicide_attempt':'suicide attempt',
                                                                       'HDRS_i15':'HDRS item 15: hypochondriasis',
                                                                       'fam_history':'psychiatric family history',
                                                                       's_nat_german':'native German speaker',
                                                                       'HDRS_i16':'HDRS item 16: loss of weight',
                                                                       'SCL_pho_anx':'SCL-90-R: phobic anxiety',
                                                                       'SLP':'currently taking sleep medication',
                                                                       'TCA':'currently taking tricyclic antidepressants',
                                                                       'HDRS_i06':'HDRS item 6: insomnia: early hours of the morning',
                                                                       'HDRS_i20':'HDRS item 20: paranoid symptoms',
                                                                       'violent_suicide_attempt':'violent suicide attempts in medical history',
                                                                       'NL':'currently taking antipsychotics',
                                                                       'school_edu':'school education',
                                                                       'SCL_dep':'SCL-90-R: depression',
                                                                       'HDRS_i09':'HDRS item 9: agitation',
                                                                       'HDRS17_total':'HDRS-17: total score',
                                                                       'SCL_int_sen':'SCL-90-R: interpersonal sensitivity',
                                                                       'years_since_first_hos':'time since first hospitalization',
                                                                       'HDRS_veg_dep':'HDRS: vegetative depression subscale',
                                                                       'SCL_GSI':'SCL-90-R: global severity',
                                                                       'SCL_anx':'SCL-90-R: anxiety',
                                                                       'job_edu':'professional education'})

    p = (
    ggplot(importances, aes(x = 'feature_name', y = 'importance', color = 'model', shape = 'model')) +
    geom_hline(yintercept = 0, linetype = '--', color = 'grey', alpha = 0.75) +
    stat_summary(fun_data = 'mean_cl_normal', fun_args = {'mult':1}, geom = 'errorbar', alpha = 0.75, position = position_dodge(width = 0.8)) +
    stat_summary(fun_y = np.mean, geom = 'point', alpha = 0.75, position = position_dodge(width = 0.8)) +
    scale_x_discrete(name = '', limits = importances['feature_name'].unique().tolist()[::-1]) +
    scale_y_continuous(name = 'Permutation importance (M ± 95% CI)') +
    scale_color_manual(name = '', values = ['#1C4774', '#4075AD']) +
    coord_flip() +
    facet_grid('~classifier') +
    theme_bw() +
    theme(axis_title_x = element_text(size = 12),
          axis_text_x = element_text(size = 10, angle = 45),
          axis_title_y = element_text(size = 12),
          axis_text_y = element_text(size = 10),
          strip_text_x = element_text(size = 10),
          legend_title = element_blank())
    )
    if save:
            ggsave(p, filename = f'plots\\features_importances_clin_paper.png', height = h, width = w, units = 'cm', dpi = 900)
    return p
