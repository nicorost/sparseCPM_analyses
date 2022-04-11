# -----------------------------------------------------------------------------
#                             FUNCTION FOR ML PIPELINE
# -----------------------------------------------------------------------------
# (C) Nicolas Rost, 2022

import time
import random
import numpy as np
import multiprocessing
import scipy.stats as stats
from skopt import BayesSearchCV
from sklearn.svm import LinearSVC
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import loguniform
from sklearn.feature_selection import RFECV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold



def start_pipeline(features,
                   Xtrain,
                   Xtest,
                   ytrain,
                   ytest,
                   classifier,
                   rfe = True,
                   n_folds = 5,
                   n_repeats = 25,
                   n_params = 100,
                   permutations = False):
    '''
    Function for proposed predictive modeling pipeline.

    Specify parameters via inputs:
    1) which sklearn classifier should be used (SGDClassifier, RandomForestClassifier, SVC)
    2) if recursive feature elimination should be used
    3) number of folds for inner and/or outer CV
    4) number of repeats of CV
    5) number of hyperparameter combinations for random search
    6) if target should be permuted or not
    '''

    # SOME SAFETY CHECKS
    allowed_clfs = ['sgdc', 'rfcl', 'svc']
    if classifier not in allowed_clfs:
        raise ValueError("Unknown classifier entered. Please select one of the following: {}".format(allowed_clfs))


    # PRESPECIFICATIONS
    rs = 1897 # random_state/seed
    max_cores = multiprocessing.cpu_count()
    rskfold = RepeatedStratifiedKFold(n_splits = n_folds, n_repeats = n_repeats, random_state = rs)
    scl = StandardScaler()
    imp = KNNImputer(n_neighbors = 5, weights = 'uniform')
    sgdc = SGDClassifier(loss = 'log', penalty = 'elasticnet', class_weight = 'balanced', random_state = rs)
    rfcl = RandomForestClassifier(class_weight = 'balanced', random_state = rs)
    svc = LinearSVC(dual = False, max_iter = 1e5, class_weight = 'balanced', random_state = rs)

    print('\nFramework:')
    if rfe == True:
        print('\nRepeated nested cross validation with Bayesian search and recursive feature elimination.')
    else:
        print('\nRepeated cross validation with Bayesian search.')
    print('Number of folds: {}'.format(n_folds))
    print('Repeats of outer CV: {}'.format(n_repeats))
    print('Number of sampled parameter combinations: {}'.format(n_params))
    if permutations > 0:
        print('Number of permutations: {}'.format(permutations))
    print('Classifier: {}'.format(classifier))


    # CUSTOM PIPELINE
    # passing coef amd importance through pipe (doesn't work with standard pipeline)
    class MyPipeline(Pipeline):
        @property
        def coef_(self):
            return self._final_estimator.coef_
        @property
        def feature_importances_(self):
            return self._final_estimator.feature_importances_


    # PIPELINES AND PARAMETERS FOR CLASSIFIERS
    if classifier == 'sgdc':
        clf = sgdc
        # preprocssing pipeline (scaling and imputing)
        preproc = Pipeline([('scl', scl),
                            ('imp', imp)])
        # create hyperparameter space
        if rfe == True:
            param_bayes = {'rfecv__estimator__clf__l1_ratio': Real(0, 1, prior = 'uniform'),
                           'rfecv__estimator__clf__alpha': Real(0.001, 1, prior = 'log-uniform')}
        else:
            param_bayes = {'clf__l1_ratio': Real(0, 1, prior = 'uniform'),
                           'clf__alpha': Real(0.001, 1, prior = 'log-uniform')}
        print('\nParameter Distributions:\n{}'.format(param_bayes))

    if classifier == 'rfcl':
        clf = rfcl
        # preprocssing pipeline (imputing)
        preproc = Pipeline([('imp', imp)])
        # create hyperparameter space
        if rfe == True:
            param_bayes = {'rfecv__estimator__clf__n_estimators': Integer(50, 150),
                           'rfecv__estimator__clf__max_features': Categorical(['auto', 'log2']),
                           'rfecv__estimator__clf__max_depth': Integer(2, 3),
                           'rfecv__estimator__clf__min_samples_leaf': Integer(12, 16)}
        else:
            param_bayes = {'clf__n_estimators': Integer(50, 150),
                           'clf__max_features': Categorical(['auto', 'log2']),
                           'clf__max_depth': Integer(2, 3),
                           'clf__min_samples_leaf': Integer(12, 16)}
        print('\nParameter Distributions:\n{}'.format(param_bayes))

    if classifier == 'svc':
        clf = svc
        # preprocssing pipeline (scaling and imputing)
        preproc = Pipeline([('scl', scl),
                            ('imp', imp)])
        # create hyperparameter space
        if rfe == True:
            param_bayes = {'rfecv__estimator__clf__C': Real(1e-5, 1e3, prior = 'log-uniform')}
        else:
            param_bayes = {'clf__C': Real(1e-5, 1e3, prior = 'log-uniform')}
        print('\nParameter Distributions:\n{}'.format(param_bayes))


    # START PIPELINES WITHOUT PERMUTATIONS
    if permutations == False:

        if rfe == True:
            # inner CV pipeline
            inner_pipe = MyPipeline([('preproc', preproc),
                                     ('clf', clf)])
            rfecv = RFECV(inner_pipe, cv = n_folds)
            # outer CV pipeline
            outer_pipe = Pipeline([('preproc', preproc),
                                   ('rfecv', rfecv)])

            print('\nStart computation on {} available cores...\n'.format(max_cores))
            start_time = time.time()
            rskfold_search = BayesSearchCV(outer_pipe, param_bayes, n_iter = n_params, cv = rskfold, scoring = 'balanced_accuracy', return_train_score = True, random_state = rs, verbose = 1, n_jobs = -1)
            rskfold_search.fit(Xtrain, ytrain)
            print('\n--- Computation took {} minutes ---'.format((time.time() - start_time) / 60))

            # evaluation
            print('\nBest parameters: {}'.format(rskfold_search.best_params_)) # best parameters from training set
            print('\nMean cross-validated score of best estimator: {:.3f}'.format(rskfold_search.best_score_)) # mean accuracy of best parameters over CV
            print('\nBest estimator:\n{}'.format(rskfold_search.best_estimator_.steps[1][1].estimator)) # print out best estimator
            # selected features from best estimator
            n_features = rskfold_search.best_estimator_.steps[1][1].n_features_
            print('\nBest number of features from RFE: {}'.format(n_features)) # results from recursive feature elimination
            selected_features = features.columns.values[rskfold_search.best_estimator_.steps[1][1].support_]
            print('\nSelected features from RFE:\n{}'.format(selected_features))

        else:
            pipe = Pipeline([('preproc', preproc),
                             ('clf', clf)])
            print('\nStart computation on {} available cores...\n'.format(max_cores))
            start_time = time.time()
            rskfold_search = BayesSearchCV(pipe, param_bayes, n_iter = n_params, cv = rskfold, scoring = 'balanced_accuracy', return_train_score = True, random_state = rs, verbose = 1, n_jobs = -1)
            rskfold_search.fit(Xtrain, ytrain)
            print('\n--- Computation took {} minutes ---'.format((time.time() - start_time) / 60))

            # evaluation
            print('\nBest parameters: {}'.format(rskfold_search.best_params_)) # best parameters from training set
            print('\nMean cross-validated score of best estimator: {:.3f}'.format(rskfold_search.best_score_)) # mean accuracy of best parameters over CV
            print('\nBest estimator:\n{}'.format(rskfold_search.best_estimator_)) # print out best estimator
            # selected features from best estimator
            if classifier in ['sgdc', 'svc']:
                weights = rskfold_search.best_estimator_.steps[1][1].coef_
            else:
                weights = rskfold_search.best_estimator_.steps[1][1].feature_importances_
            n_features = np.sum(weights != 0)
            print('\nNumber of features: {}'.format(n_features)) # results from recursive feature elimination
            selected_features = features.columns.values[(weights != 0).reshape(-1)]
            print('\nSelected features from best estimator:\n{}'.format(selected_features))

    # START PIPELINES WITH PERMUTATIONS (AND RFECV)
    if permutations > 0:

        if rfe == False:
            raise ValueError("If permutation is not False, rfe should be True")

        rskfold_search = {} # dict with results
        selected_features = {} # dict with results
        print('\nStart computation on {} available cores...'.format(max_cores))
        start_time = time.time()
        for p in range(permutations):
            print('\nStarting permutation run {}...'.format(p + 1))
            # permute y
            ytrain_temp = np.random.RandomState(seed = p).permutation(ytrain)
            ytest_temp = np.random.RandomState(seed = p).permutation(ytest)
            # inner CV pipeline
            inner_pipe = MyPipeline([('preproc', preproc),
                                     ('clf', clf)])
            rfecv = RFECV(inner_pipe, cv = n_folds)
            # outer CV pipeline
            outer_pipe = Pipeline([('preproc', preproc),
                                   ('rfecv', rfecv)])
            rskfold_search_temp = BayesSearchCV(outer_pipe, param_bayes, n_iter = n_params, cv = rskfold, scoring = 'balanced_accuracy', return_train_score = True, random_state = rs, n_jobs = -1)
            rskfold_search_temp.fit(Xtrain, ytrain_temp)

            # evaluation
            rskfold_search[p] = {}
            rskfold_search[p]['best_params'] = rskfold_search_temp.best_params_
            rskfold_search[p]['best_score'] = rskfold_search_temp.best_score_
            rskfold_search[p]['best_estimator'] = rskfold_search_temp.best_estimator_
            rskfold_search[p]['cv_results'] = rskfold_search_temp.cv_results_
            rskfold_search[p]['test_score'] = rskfold_search_temp.score(Xtest, ytest_temp)

            selected_features[p] = {}
            selected_features[p]['n_features'] = rskfold_search_temp.best_estimator_.steps[1][1].n_features_
            selected_features[p]['features'] = features.columns.values[rskfold_search_temp.best_estimator_.steps[1][1].support_]

        print('\n--- Computation took {} minutes ---'.format((time.time() - start_time) / 60))


    return rskfold_search, selected_features
