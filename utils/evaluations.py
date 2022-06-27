# --------------------------------------------------------------------------
#        EVALUATION FUNCTION FOR ML PIPELINE RESULTS
# --------------------------------------------------------------------------
# (C) Nicolas Rost, 2022

import pickle
import numpy as np
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, plot_roc_curve
from utils.performances import model_performance
from utils.data_simulation import create_simulated_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D
from scipy.stats import beta, kstest
import statsmodels.api as sm



def calcuate_performances(Xtrain, ytrain, Xtest, ytest):
    """
    Function that calculates and saves the performance values (MCC, BAC) from all models.

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

    Returns
    -------
    None.

    """

    # recreate simulation data
    X_sim, y_sim = create_simulated_data()
    Xtrain_sim, Xtest_sim, ytrain_sim, ytest_sim = train_test_split(X_sim, y_sim, test_size = 0.2, random_state = 1897)

    # data types and classifiers for loop
    data = ['clin_sgdc', 'sim_sgdc', 'clin_rfcl', 'sim_rfcl', 'clin_svc', 'sim_svc']

    # dicts for saving
    model_performances = {}
    model_performances['MCC'] = {}
    model_performances['BAC'] = {}

    # load data
    for d in data:
        f = open(f'results/results_pipeline_{d}_no_rfe', 'rb')
        res_pipe = pickle.load(f)
        f.close()
        f = open(f'results/results_pipeline_{d}_rfe', 'rb')
        res_pipe_rfe = pickle.load(f)
        f.close()
        f = open(f'results/results_pipeline_{d}_rfe_permutations', 'rb')
        res_pipe_perms = pickle.load(f)
        f.close()

        # performances
        # no rfe
        m = res_pipe.best_estimator_
        if 'clin' in d:
            m.fit(Xtrain, ytrain)
            ypred = m.predict(Xtest)
            model_performances['BAC'][f'{d}_no_rfe'] = balanced_accuracy_score(ytest, ypred)
            model_performances['MCC'][f'{d}_no_rfe'] = matthews_corrcoef(ytest, ypred)
        if 'sim' in d:
            m.fit(Xtrain_sim, ytrain_sim)
            ypred_sim = m.predict(Xtest_sim)
            model_performances['BAC'][f'{d}_no_rfe'] = balanced_accuracy_score(ytest_sim, ypred_sim)
            model_performances['MCC'][f'{d}_no_rfe'] = matthews_corrcoef(ytest_sim, ypred_sim)
        # rfe
        m_rfe = res_pipe_rfe.best_estimator_.steps[1][1].estimator_
        if 'clin' in d:
            Xtrain_red = Xtrain[:, res_pipe_rfe.best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
            Xtest_red = Xtest[:, res_pipe_rfe.best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
            m_rfe.fit(Xtrain_red, ytrain)
            ypred = m_rfe.predict(Xtest_red)
            model_performances['BAC'][f'{d}_rfe'] = balanced_accuracy_score(ytest, ypred)
            model_performances['MCC'][f'{d}_rfe'] = matthews_corrcoef(ytest, ypred)
        if 'sim' in d:
            Xtrain_sim_red = Xtrain_sim[:, res_pipe_rfe.best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
            Xtest_sim_red = Xtest_sim[:, res_pipe_rfe.best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
            m_rfe.fit(Xtrain_sim_red, ytrain_sim)
            ypred_sim = m_rfe.predict(Xtest_sim_red)
            model_performances['BAC'][f'{d}_rfe'] = balanced_accuracy_score(ytest_sim, ypred_sim)
            model_performances['MCC'][f'{d}_rfe'] = matthews_corrcoef(ytest_sim, ypred_sim)
        # for permutations
        perms = 100
        model_performances['BAC'][f'{d}_rfe_permutations'] = []
        model_performances['MCC'][f'{d}_rfe_permutations'] = []
        for p in range(perms):
            m_perms = res_pipe_perms[p]['best_estimator'].steps[1][1].estimator_
            if 'clin' in d:
                ytrain_temp = np.random.RandomState(seed = p).permutation(ytrain)
                ytest_temp = np.random.RandomState(seed = p).permutation(ytest)
                Xtrain_red = Xtrain[:, res_pipe_perms[p]['best_estimator'].steps[1][1].support_]
                Xtest_red = Xtest[:, res_pipe_perms[p]['best_estimator'].steps[1][1].support_]
                m_perms.fit(Xtrain_red, ytrain_temp)
                ypred = m_perms.predict(Xtest_red)
                model_performances['BAC'][f'{d}_rfe_permutations'].append(balanced_accuracy_score(ytest_temp, ypred))
                model_performances['MCC'][f'{d}_rfe_permutations'].append(matthews_corrcoef(ytest_temp, ypred))
            if 'sim' in d:
                ytrain_sim_temp = np.random.RandomState(seed = p).permutation(ytrain_sim)
                ytest_sim_temp = np.random.RandomState(seed = p).permutation(ytest_sim)
                Xtrain_sim_red = Xtrain_sim[:, res_pipe_perms[p]['best_estimator'].steps[1][1].support_]
                Xtest_sim_red = Xtest_sim[:, res_pipe_perms[p]['best_estimator'].steps[1][1].support_]
                m_perms.fit(Xtrain_sim_red, ytrain_sim_temp)
                ypred_sim = m_perms.predict(Xtest_sim_red)
                model_performances['BAC'][f'{d}_rfe_permutations'].append(balanced_accuracy_score(ytest_sim_temp, ypred_sim))
                model_performances['MCC'][f'{d}_rfe_permutations'].append(matthews_corrcoef(ytest_sim_temp, ypred_sim))

    f = open('results/model_performances', 'wb')
    pickle.dump(model_performances, f)
    f.close()

    return model_performances


def evaluate_pipeline(data,
                      clf,
                      Xtrain, ytrain, Xtest, ytest,
                      save = False):
    """
    Function to evaluate predictive modeling pipeline with RFE, without RFE and
    with RFE and permutations.

    Parameters
    ----------
    data : STR
        Has to be 'clin' or 'sim'.
    clf : STR
        Has to be 'sgdc', 'rfcl', or 'svc'.
    Xtrain : np.array
        Clinical training features. Simulation training data will be created in here again.
    ytrain : np.array
        Clinical training target. Simulation training data will be created in here again.
    Xtest : np.array
        Clinical test features. Simulation test data will be created in here again.
    ytest : np.array
        Clinical test target. Simulation test data will be created in here again.
    save : BOOL, optional
        Should the plots be saved. The default is False.

    Returns
    -------
    All three objects for the three different methods.

    """

    # load all three methods (rfe, no rfe, permutations)
    f = open(f'results/results_pipeline_{data}_{clf}_no_rfe', 'rb')
    res_pipe = pickle.load(f)
    f.close()
    f = open(f'results/results_pipeline_{data}_{clf}_rfe', 'rb')
    res_pipe_rfe = pickle.load(f)
    f.close()
    f = open(f'results/results_pipeline_{data}_{clf}_rfe_permutations', 'rb')
    res_pipe_perms = pickle.load(f)
    sel_feat_perms = pickle.load(f)
    f.close()

    # if simulated data is selected, recreate it here again
    if data == 'sim':
        X, y = create_simulated_data()
        indices = range(X.shape[0])
        Xtrain, Xtest, ytrain, ytest, train_idx, test_idx = train_test_split(X, y, indices, test_size = 0.2, random_state = 1897)

    # test set performances
    # no rfe
    print('\nResults from no RFE:')
    m = res_pipe.best_estimator_
    m.fit(Xtrain, ytrain)
    ypred = m.predict(Xtest)
    bac = balanced_accuracy_score(ytest, ypred)
    mcc = matthews_corrcoef(ytest, ypred)
    if clf != 'svc':
        model_performance(m, Xtrain, ytrain, Xtest, ytest, 'test')
    else:
        model_performance(m, Xtrain, ytrain, Xtest, ytest, 'test', plot_auc = False)
    # rfe
    print('\nResults from RFE:')
    m_rfe = res_pipe_rfe.best_estimator_.steps[1][1].estimator_
    Xtrain_red = Xtrain[:, res_pipe_rfe.best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
    Xtest_red = Xtest[:, res_pipe_rfe.best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
    m_rfe.fit(Xtrain_red, ytrain)
    ypred = m_rfe.predict(Xtest_red)
    bac_rfe = balanced_accuracy_score(ytest, ypred)
    mcc_rfe = matthews_corrcoef(ytest, ypred)
    if clf != 'svc':
        model_performance(m_rfe, Xtrain_red, ytrain, Xtest_red, ytest, 'test')
    else:
        model_performance(m_rfe, Xtrain_red, ytrain, Xtest_red, ytest, 'test', plot_auc = False)
    # for permutations
    perms = 100
    bac_scores = []
    mcc_scores = []
    n_features = []
    for p in range(perms):
        ytrain_temp = np.random.RandomState(seed = p).permutation(ytrain)
        ytest_temp = np.random.RandomState(seed = p).permutation(ytest)
        m_perms = res_pipe_perms[p]['best_estimator'].steps[1][1].estimator_
        Xtrain_red = Xtrain[:, res_pipe_perms[p]['best_estimator'].steps[1][1].support_] # reduced feature set selected by RFE
        Xtest_red = Xtest[:, res_pipe_perms[p]['best_estimator'].steps[1][1].support_] # reduced feature set selected by RFE
        m_perms.fit(Xtrain_red, ytrain_temp)
        ypred = m_perms.predict(Xtest_red)
        bac_scores.append(balanced_accuracy_score(ytest_temp, ypred))
        mcc_scores.append(matthews_corrcoef(ytest_temp, ypred))
        n_features.append(sel_feat_perms[p]['n_features'])

    # plot all three models
    sns.set(style = 'whitegrid', font_scale = 1.5)
    legend_elements = [Line2D([0], [0], color = '#76A29F', lw = 2, label = 'permutations'),
                       Line2D([0], [0], color = '#FEB302', lw = 2, label = 'no RFE', linestyle = '--'),
                       Line2D([0], [0], color = '#FF5D3E', lw = 2, label = 'RFE')]
    # 1. balanced accuracies
    sns.displot(bac_scores, alpha = 0.5, rug = True, kde = True, bins = 20, color = '#76A29F', aspect = 1.45)
    plt.annotate(f'{np.mean(bac_scores):.2f}', (np.mean(bac_scores), 0.825), xycoords = ('data', 'figure fraction'), color = '#76A29F')
    plt.axvline(bac, 0, 0.75, linestyle = '--', alpha = 0.75, color = '#FEB302')
    plt.axvline(bac_rfe, 0, 0.75, linestyle = '-', alpha = 0.75, color = '#FF5D3E')
    plt.annotate(f'{bac:.2f}', (bac, 0.8), xycoords = ('data', 'figure fraction'), color = '#FEB302')
    plt.annotate(f'{bac_rfe:.2f}', (bac_rfe, 0.85), xycoords = ('data', 'figure fraction'), color = '#FF5D3E')
    plt.xticks(np.arange(round(min(bac_scores), 1) - 0.1, round(max(max(bac_scores), max([bac]), max([bac_rfe])), 1) + 0.1, 0.1))
    plt.xlabel('Balanced accuracy')
    plt.legend(handles = legend_elements, bbox_to_anchor = (1.42, 1))
    if save:
        plt.savefig(f'plots\BACs_{data}_{clf}.png', dpi = 900, bbox_inches = 'tight')
    plt.show()
    # 2. Matthew's correlation coefficient
    sns.displot(mcc_scores, alpha = 0.5, rug = True, kde = True, bins = 20, color = '#76A29F', aspect = 1.45)
    plt.annotate(f'{np.mean(mcc_scores):.2f}', (np.mean(mcc_scores), 0.825), xycoords = ('data', 'figure fraction'), color = '#76A29F')
    plt.axvline(mcc, 0, 0.75, linestyle = '--', alpha = 0.75, color = '#FEB302')
    plt.axvline(mcc_rfe, 0, 0.75, linestyle = '-', alpha = 0.75, color = '#FF5D3E')
    plt.annotate(f'{mcc:.2f}', (mcc, 0.8), xycoords = ('data', 'figure fraction'), color = '#FEB302')
    plt.annotate(f'{mcc_rfe:.2f}', (mcc_rfe, 0.85), xycoords = ('data', 'figure fraction'), color = '#FF5D3E')
    plt.xticks(np.arange(round(min(mcc_scores), 1) - 0.1, round(max(max(mcc_scores), max([mcc]), max([mcc_rfe])), 1) + 0.1, 0.1))
    plt.xlabel("Matthews correlation coefficient")
    plt.legend(handles = legend_elements, bbox_to_anchor = (1.42, 1))
    if save:
        plt.savefig(f'plots\MCCs_{data}_{clf}.png', dpi = 900, bbox_inches = 'tight')
    plt.show()


    # number of features
    if clf in ['sgdc', 'svc']:
        weights = res_pipe.best_estimator_.steps[1][1].coef_
        weights_rfe = res_pipe_rfe.best_estimator_.steps[1][1].estimator_.steps[1][1].coef_
    else:
        weights = res_pipe.best_estimator_.steps[1][1].feature_importances_
        weights_rfe = res_pipe_rfe.best_estimator_.steps[1][1].estimator_.steps[1][1].feature_importances_

    n_features = np.sum(weights != 0)
    n_features_rfe = np.sum(weights_rfe != 0)

    print(f'\nNumber of features for model without RFE: {n_features}')
    print(f'\nNumber of features for model with RFE: {n_features_rfe}')


    return res_pipe, res_pipe_rfe, res_pipe_perms


def plot_performances(metric,
                      Xtrain, ytrain, Xtest, ytest,
                      t_dis = False, save = False):
    """
    Function to plot the model performances, ordered by dataset and classifier.

    Parameters
    ----------
    metric : STR
        Enter 'BAC' or 'MCC'
    Xtrain : np.array
        Clinical training features. Simulation training data will be created in here again.
    ytrain : np.array
        Clinical training target. Simulation training data will be created in here again.
    Xtest : np.array
        Clinical test features. Simulation test data will be created in here again.
    ytest : np.array
        Clinical test target. Simulation test data will be created in here again.
    t_dis : BOOL, optional
        Whether the MCC plots should include the theoretical t-distribution over the permutations
    save : BOOL, optional
        Should the plot be saved. The default is False.

    Returns
    -------
    Plot of performances.
    """

    # recreate simulation data
    X_sim, y_sim = create_simulated_data()
    Xtrain_sim, Xtest_sim, ytrain_sim, ytest_sim = train_test_split(X_sim, y_sim, test_size = 0.2, random_state = 1897)

    # load performances and select metric
    f = open('results/model_performances', 'rb')
    model_performances = pickle.load(f)
    f.close()
    res = model_performances[metric]

    # create subplots
    data = ['clin_sgdc', 'sim_sgdc', 'clin_rfcl', 'sim_rfcl', 'clin_svc', 'sim_svc']
    sns.set(style = 'whitegrid', font_scale = 1.5)
    fig, axes = plt.subplots(3, 2, figsize = (15, 15))
    # legend and shared x- & y-labels
    legend_elements = [Line2D([0], [0], color = '#76A29F', lw = 2, label = 'permutations', ls = ':'),
                       Line2D([0], [0], color = '#FEB302', lw = 2, label = 'no RFE', ls = '--'),
                       Line2D([0], [0], color = '#FF5D3E', lw = 2, label = 'RFE')]
    ylabels = ['LR', 'RF', 'SVC']
    xlabels = ['Clinical data', 'Simulated data']

    # loop through results and create plots
    for i, ax in enumerate(axes.flatten()):

        # performances
        d = data[i]
        perf = res[f'{d}_no_rfe']
        perf_rfe = res[f'{d}_rfe']
        perf_scores = res[f'{d}_rfe_permutations']

        # create plots
        sns.histplot(perf_scores, alpha = 0.5, kde = True, stat = 'density', bins = 20, color = '#76A29F', line_kws = {'ls':':', 'lw':2}, ax = ax)
        sns.rugplot(perf_scores, alpha = 0.5, color = '#76A29F', ax = ax)
        ax.annotate(f'{np.mean(perf_scores):.2f}', (np.mean(perf_scores), 0.825), xycoords = ('data', 'axes fraction'), color = '#76A29F')
        ax.axvline(perf, 0, 0.75, ls = '--', lw = 2, alpha = 0.75, color = '#FEB302')
        ax.axvline(perf_rfe, 0, 0.75, ls = '-', lw = 2, alpha = 0.75, color = '#FF5D3E')
        ax.annotate(f'{perf:.2f}', (perf, 0.8), xycoords = ('data', 'axes fraction'), color = '#FEB302')
        ax.annotate(f'{perf_rfe:.2f}', (perf_rfe, 0.875), xycoords = ('data', 'axes fraction'), color = '#FF5D3E')
        ax.set_xlim([round(min(perf_scores), 1) - 0.1, round(max(max(perf_scores), max([perf]), max([perf_rfe])), 1) + 0.1])
        ax.set_xticks(np.arange(round(min(perf_scores), 1) - 0.1, round(max(max(perf_scores), max([perf]), max([perf_rfe])), 1) + 0.1, 0.1))
        ax.set_ylabel('')
        if (metric == 'MCC') & (t_dis == True):
            n = Xtest.shape[0]
            x = np.linspace(-1, 1, 100)
            dist = beta(n/2-1, n/2-1, loc=-1, scale=2)
            ax.plot(x, dist.pdf(x), color = 'grey', linestyle = ':')

    if metric == 'BAC':
        metric_label = 'Balanced accuracy'
    if metric == 'MCC':
        metric_label = 'Matthews correlation coefficient'
    for ax, col_label in zip(axes[0], xlabels):
        ax.set_title(col_label, y = 1.05)
    for ax in axes[2]:
        ax.set_xlabel(metric_label)
    for ax, row_label in zip(axes[:,0], ylabels):
        ax.set_ylabel(f"{row_label}\n\nDensity", rotation = 90)
    # if metric == 'BAC':
    #     plt.annotate('Balanced accuracy', (0.5, 0.04), xycoords = 'figure fraction', ha = 'center')
    # if metric == 'MCC':
    #     plt.annotate('Matthews correlation coefficient', (0.5, 0.03), xycoords = 'figure fraction', ha = 'center')
    # plt.annotate('Density', (0.03, 0.475), xycoords = 'figure fraction', va = 'center', rotation = 'vertical')
    plt.legend(handles = legend_elements, bbox_to_anchor = (1.075, 0.5), bbox_transform = fig.transFigure)
    if save:
        plt.savefig(f'plots\{metric}_all_models.png', dpi = 300, bbox_inches = 'tight')
    plt.show()


def qq_plot_permutations(Xtrain, ytrain, Xtest, ytest,
                         save = False):
    """
    Function to QQ plot the permutations (MCCs) against the theoretical distribution.

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
    Plot of performances.
    """

    # recreate simulation data
    X_sim, y_sim = create_simulated_data()
    Xtrain_sim, Xtest_sim, ytrain_sim, ytest_sim = train_test_split(X_sim, y_sim, test_size = 0.2, random_state = 1897)

    # load performances and select metric
    f = open('results/model_performances', 'rb')
    model_performances = pickle.load(f)
    f.close()
    res = model_performances['MCC']

    # dict for Kolmogorov-Smirnov tests
    ks_tests = {}

    # create subplots
    data = ['clin_sgdc', 'sim_sgdc', 'clin_rfcl', 'sim_rfcl', 'clin_svc', 'sim_svc']
    sns.set(style = 'whitegrid', font_scale = 1.5)
    fig, axes = plt.subplots(3, 2, figsize = (15, 15))
    ylabels = ['LR', 'RF', 'SVC']
    xlabels = ['Clinical data', 'Simulated data']

    # loop through results and create plots
    for i, ax in enumerate(axes.flatten()):

        # load data
        d = data[i]
        perf_scores = res[f'{d}_rfe_permutations']

        # Kolmogorov-Smirnov tests
        if 'clin' in d:
            n = Xtest.shape[0]
        if 'sim' in d:
            n = Xtest_sim.shape[0]
        ks_test = kstest(perf_scores, 'beta', args = (n/2-1, n/2-1, -1, 2))
        ks_tests[d] = ks_test

        # theoretical distribution
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        dist = beta(n/2-1, n/2-1, loc = -1, scale = 2)

        # QQ-plot
        sm.qqplot(np.array(perf_scores), dist, line = '45', ax = ax)
        ax.get_lines()[0].set_markerfacecolor('#76A29F')
        ax.get_lines()[0].set_markeredgecolor('#76A29F')
        ax.get_lines()[0].set_alpha(0.75)
        ax.get_lines()[1].set_color("gray")
        ax.get_lines()[1].set_linestyle('--')
        ax.set_xlabel('')
        ax.set_ylabel('')

    for ax, col_label in zip(axes[0], xlabels):
        ax.set_title(col_label, y = 1.05)
    for ax in axes[2]:
        ax.set_xlabel('Theoretical Quantiles')
    for ax, row_label in zip(axes[:,0], ylabels):
        ax.set_ylabel(f'{row_label}\n\nSample Quantiles', rotation = 90)

    if save:
        plt.savefig(f'plots\QQ_plots_permutations_MCC.png', dpi = 900, bbox_inches = 'tight')
    plt.show()

    return ks_tests


def plot_roc_curves(Xtrain, ytrain, Xtest, ytest,
                    save = False):
    """
    Function to plot the ROC curves of the non-permuted models.

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
    Plot of performances.
    """

    # recreate simulation data
    X_sim, y_sim = create_simulated_data()
    Xtrain_sim, Xtest_sim, ytrain_sim, ytest_sim = train_test_split(X_sim, y_sim, test_size = 0.2, random_state = 1897)

    # load all results except for permutations
    files = [f for f in Path.cwd().glob("results/results_pipeline_*") if not "permutations" in f.name]
    results = {}
    for f in files:
        temp = open(f, 'rb')
        name = str(f.name).replace("results_pipeline_", "")
        results[name] = pickle.load(temp)

    # create subplots
    data = ['clin_sgdc', 'sim_sgdc', 'clin_rfcl', 'sim_rfcl', 'clin_svc', 'sim_svc']
    sns.set(style = 'whitegrid', font_scale = 1.5)
    fig, axes = plt.subplots(3, 2, figsize = (15, 15))
    ylabels = ['LR', 'RF', 'SVC']
    xlabels = ['Clinical data', 'Simulated data']
    legend_labels = ['no RFE', 'RFE']

    # loop through results and create plots
    for i, ax in enumerate(axes.flatten()):

        # performances
        d = data[i]
        if 'clin' in d:
            # no rfe
            m = results[f'{d}_no_rfe'].best_estimator_
            m.fit(Xtrain, ytrain)
            # rfe
            m_rfe = results[f'{d}_rfe'].best_estimator_.steps[1][1].estimator_
            Xtrain_red = Xtrain[:, results[f'{d}_rfe'].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
            Xtest_red = Xtest[:, results[f'{d}_rfe'].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
            m_rfe.fit(Xtrain_red, ytrain)
            # create plots
            plot_roc_curve(m, Xtest, ytest, color = '#FEB302', name = 'no RFE', ax = ax)
            plot_roc_curve(m_rfe, Xtest_red, ytest, color = '#FF5D3E', name = 'RFE', ax = ax)
        if 'sim' in d:
            # no rfe
            m = results[f'{d}_no_rfe'].best_estimator_
            m.fit(Xtrain_sim, ytrain_sim)
            # rfe
            m_rfe = results[f'{d}_rfe'].best_estimator_.steps[1][1].estimator_
            Xtrain_sim_red = Xtrain_sim[:, results[f'{d}_rfe'].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
            Xtest_sim_red = Xtest_sim[:, results[f'{d}_rfe'].best_estimator_.steps[1][1].support_] # reduced feature set selected by RFE
            m_rfe.fit(Xtrain_sim_red, ytrain_sim)
            # create plots
            plot_roc_curve(m, Xtest_sim, ytest_sim, color = '#FEB302', name = 'no RFE', ax = ax)
            plot_roc_curve(m_rfe, Xtest_sim_red, ytest_sim, color = '#FF5D3E', name = 'RFE', ax = ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
    for ax, col_label in zip(axes[0], xlabels):
        ax.set_title(col_label, y = 1.05)
    for ax in axes[2]:
        ax.set_xlabel('False positive rate')
    for ax, row_label in zip(axes[:,0], ylabels):
        ax.set_ylabel(f'{row_label}\n\nTrue positive rate', rotation = 90)
    if save:
        plt.savefig('plots\\roc_curves_all_models.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
