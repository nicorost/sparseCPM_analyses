# -----------------------------------------------------------------------------
#                      PREDICTIVE MODELING PIPELINE
#                      REPEATED NESTED CV WITH RFE
# -----------------------------------------------------------------------------
# (C) Nicolas Rost, 2022

print('Starting script.')

# ----------------------------- IMPORT PACKAGES -------------------------------
import os
os.chdir(r'C:\Users\nicolas_rost\Documents\_PhD_MPI\devoted\manuscripts\methods_paper\analysis') # set path
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from tableone import TableOne
import utils.evaluations as ue
from argparse import ArgumentParser
from pipeline import start_pipeline
from preproc import preprocess_data
import utils.feature_selections as uf
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from utils.data_simulation import create_simulated_data
from utils.info_clinical_features import summarize_clinical_info


# ----------------------- ARGUMENTS FROM COMMAND LINE -------------------------
parser = ArgumentParser(description = 'ML Pipeline for methods paper.')
parser.add_argument('-data', default = 'clin', help = 'select "clin" or "sim"')
parser.add_argument('-clf', default = 'sgdc', help = 'select classifier')
parser.add_argument('-rfe', default = 'yes', help = 'RFE: "yes" or "no"?')
parser.add_argument('-perms', default = False, help = 'Number of permutations', type = int)
args = parser.parse_args()


# ------------------------ IMPORT OR SIMULATE DATA ----------------------------
# for clinical data
if args.data == 'clin':
    print('\nClinical data selected.')
    mars = pd.read_pickle('MARS_preprocessed_ML_PRScs.pkl')
    mars = mars.drop('NID', axis = 1)
    # additional preprocessing
    features, outcome = preprocess_data(data = mars,
                                        target = 'KRS17me_6W',
                                        na_cutoff_samples = 0.75,
                                        na_cutoff_features = 0.30,
                                        min_count_binary = 0.05,
                                        include_bio = False)
    # convert into arrays
    X = features.values
    y = outcome.values.ravel() * (-1) # -1 if KRS17me_6W is the target!
# for simluated data
if args.data == 'sim':
    print('\nSimulated data selected.')
    X, y = create_simulated_data()
    # additionally, we need to create arbitrary feature names for importances below
    features = pd.DataFrame(X, columns = [f"feature_{var + 1:03d}" for var in range(X.shape[1])])


# -------------------- CREATE TARINING AND TEST SET ---------------------------
indices = range(X.shape[0])
Xtrain, Xtest, ytrain, ytest, train_idx, test_idx = train_test_split(X, y, indices, test_size = 0.2, random_state = 1897)
print('\nFinal number of features: {}'.format(Xtrain.shape[1]))
print('\nFinal number of samples:\nTraining: {}\nTest: {}'.format(Xtrain.shape[0], Xtest.shape[0]))


# -------------------- DESCRIPTIVES TABLE FOR CLINICAL DATA -------------------
df_desc = features.join(outcome)
df_desc['set'] = 'training'
df_desc['set'].iloc[test_idx] = 'test'
columns = ['age', 'sex_female', 'icd_diag_F33', 'HMD17_00', 'KRS17me_6W']
desc = TableOne(df_desc, columns = columns, groupby = 'set', pval = True)
print(desc.tabulate(tablefmt = 'github'))
# also all features with descriptions (for Suppl. Table)
df_desc = summarize_clinical_info(features, save = True)


# --------------------------- START PIPELINE ----------------------------------
rfe = True if args.rfe == "yes" else False # boolean input for argparse didn't work
res_pipe, sel_feat = start_pipeline(features,
                                    Xtrain, Xtest, ytrain, ytest,
                                    classifier = args.clf,
                                    rfe = rfe,
                                    n_folds = 5,
                                    n_repeats = 5,
                                    n_params = 100,
                                    permutations = args.perms)


# ----------------- SAVE RESULTS FROM PERMUTATIONS-----------------------------
# we need to create the MyPipeline objects again at the top level here in order to be able to pickle them
class MyPipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_
    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_

# create name for saving and explicitly specify all MyPipeline objects again
if rfe == True:
    if args.perms == False:
        save_name = f'results_pipeline_{args.data}_{args.clf}_rfe'
        res_pipe.best_estimator_.steps[1][1].estimator = MyPipeline(res_pipe.best_estimator_.steps[1][1].estimator.steps)
        res_pipe.estimator.steps[1][1].estimator = MyPipeline(res_pipe.estimator.steps[1][1].estimator.steps)
        res_pipe.best_estimator_._final_estimator.estimator = MyPipeline(res_pipe.best_estimator_._final_estimator.estimator.steps)
        res_pipe.best_estimator_._final_estimator.estimator_ = MyPipeline(res_pipe.best_estimator_._final_estimator.estimator_.steps)
    if args.perms > 0:
        save_name = f'results_pipeline_{args.data}_{args.clf}_rfe_permutations'
        for p in range(args.perms):
            res_pipe[p]['best_estimator'].steps[1][1].estimator = MyPipeline(res_pipe[p]['best_estimator'].steps[1][1].estimator.steps)
            res_pipe[p]['best_estimator']._final_estimator.estimator = MyPipeline(res_pipe[p]['best_estimator']._final_estimator.estimator.steps)
            res_pipe[p]['best_estimator']._final_estimator.estimator_ = MyPipeline(res_pipe[p]['best_estimator']._final_estimator.estimator_.steps)
else:
    save_name = f'results_pipeline_{args.data}_{args.clf}_no_rfe'
print(f'\nResults will be saved as: {save_name}')

f = open(f'results/{save_name}', 'wb')
pickle.dump(res_pipe, f)
pickle.dump(sel_feat, f)
f.close()
print('\nDone saving.')


# ----------------------------- EVALUATION ------------------------------------
# NOTE: some of the functions below require the data (X, y) as inputs
# we need to enter the clinical data, the simulated data will be created again
# within the functions
if args.data != 'clin' or X.shape[0] <= 1000:
    raise ValueError('X and y might not be the clinical data! Set args.data to "clin" and run the preprocessing incl. data split again.')
# evaluate the models and save the performances
model_performances = ue.calcuate_performances(Xtrain, ytrain, Xtest, ytest)
# call function that evaluates the results (needs to be run for each dataset and each classifier)
res_pipe, res_pipe_rfe, res_pipe_perms = ue.evaluate_pipeline('clin', 'svc', Xtrain, ytrain, Xtest, ytest, save = False)
# plot ROC curves
ue.plot_roc_curves(Xtrain, ytrain, Xtest, ytest, save = False)
# plot with summary of performances
ue.plot_performances('MCC', Xtrain, ytrain, Xtest, ytest, t_dis = False, save = False)
# QQ plots for permutations
ks_tests = ue.qq_plot_permutations(Xtrain, ytrain, Xtest, ytest, save = False)
# call function that plots the number of selected features
df_plot = uf.plot_feature_numbers(Xtrain, ytrain, Xtest, ytest, save = False)
# plot features against performances (needs to be run for each metric)
df_plot = uf.plot_features_by_performance('MCC', Xtrain, ytrain, Xtest, ytest, save = False)
# get permutation importances
importances, clin_table, sim_table, clin_imps, sim_imps = uf.get_feature_importances(features, Xtrain, ytrain, Xtest, ytest, 25, save = False)
# load data, since the permutations take a while
f = open('results/permutation_importance', 'rb')
importance = pickle.load(f)
clin_table = pickle.load(f)
sim_table = pickle.load(f)
clin_imps = pickle.load(f)
sim_imps = pickle.load(f)
f.close()
# plot permutation importances (needs to be run for each dataset)
p = uf.plot_feature_importances(clin_imps, sort_by = 'top_importance', save = False)
print(p)
# for the figure in the main paper, we run the function with clearer feature names
p = uf.plot_feature_importances_main(clin_imps, save = False)
print(p)
