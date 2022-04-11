# --------------------------------------------------------------------------
#        PRINT MODEL PERFORMANCE VALUES (CLASSIFICATION)
#                       FOR IMPORT E.G.
# --------------------------------------------------------------------------
# (C) Nicolas Rost, 2022

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from scipy import interp


def model_performance(model,
                      Xtrain,
                      ytrain,
                      Xtest,
                      ytest,
                      cm_type='train',
                      plot_auc=True):
    '''
    This function prints some performance values of a classification model.
    '''

    # print label order
    n_labels = len(set(ytrain))
    print('Label Order:', sorted(set(ytrain)))

    # confusion matrices:
    ypred_train = model.predict(Xtrain)
    cm_train = confusion_matrix(ytrain, ypred_train, labels=sorted(set(ytrain)))
    print('Confusion matrix on training data:')
    print(cm_train)
    ypred_test = model.predict(Xtest)
    cm_test = confusion_matrix(ytest, ypred_test, labels=sorted(set(ytest)))
    print('Confusion matrix on test data:')
    print(cm_test)

    # check which confusion matrix will be evaluated
    eval_types = ['train', 'test']
    if cm_type not in eval_types:
        raise ValueError('Invalid confusion matrix type. Expected one of: %s' % eval_types)

    if cm_type == 'train':
        ypred = model.predict(Xtrain)
        bac = balanced_accuracy_score(ytrain, ypred)  # best calculate here
        cm_eval = cm_train
        print('\nShowing Model Performances for Training Data:')
    else:
        ypred = model.predict(Xtest)
        bac = balanced_accuracy_score(ytest, ypred)
        cm_eval = cm_test
        print('\nShowing Model Performances for Test Data:')

    # other performace values
    # Balanced Accuracy
    print('Balanced Accuracy:', np.round(bac, 3))
    # FP, FN, TP, TN
    FP = cm_eval.sum(axis = 0) - np.diag(cm_eval)
    FN = cm_eval.sum(axis = 1) - np.diag(cm_eval)
    TP = np.diag(cm_eval)
    TN = cm_eval.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    print('Sensitivity values for each class:', np.round(TPR, 3))
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    print('Specificity values for each class:', np.round(TNR, 3))
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    print('Positive Predictive Values for each class:', np.round(PPV, 3))
    # Negative predictive value
    NPV = TN / (TN + FN)
    print('Negative Predictive Values for each class:', np.round(NPV, 3))
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    print('False Positive Rate for each class:', np.round(FPR, 3))
    # False negative rate
    FNR = FN / (TP + FN)
    print('False Negative Rate for each class:', np.round(FNR, 3))
    # False discovery rate
    FDR = FP / (TP + FP)
    print('False Discovery Rate for each class:', np.round(FDR, 3))
    # F1 scores
    if n_labels > 2:
        f1 = 2 * (TPR * PPV) / (TPR + PPV)
        print('F1 scores for each cluster:', np.round(f1, 3))
    else:
        if cm_type == 'train':
            f1 = f1_score(ytrain, ypred)
        else:
            f1 = f1_score(ytest, ypred)
        print('F1 score:', np.round(f1, 3))

    # AUROC and plot
    if plot_auc:
        if n_labels > 2:
            if cm_type == 'train':
                X = Xtrain
                y = ytrain
            else:
                X = Xtest
                y = ytest
            y = label_binarize(y, classes = sorted(set(ytrain)))
            y_score = model.predict_proba(X)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_labels):
                fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute micro-average ROC curve and ROC area
            fpr['micro'], tpr['micro'], _ = roc_curve(y.ravel(), y_score.ravel())
            roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
            # Plot
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_labels)]))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_labels):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= n_labels
            fpr['macro'] = all_fpr
            tpr['macro'] = mean_tpr
            roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
            # Plot all ROC curves
            plt.figure()
            plt.plot(fpr['micro'], tpr['micro'],
                     label = 'micro-average ROC curve (area = {0:0.2f})'.format(roc_auc['micro']),
                     color = 'deeppink', linestyle = ':', linewidth = 4)
            plt.plot(fpr['macro'], tpr['macro'],
                     label = 'macro-average ROC curve (area = {0:0.2f})'.format(roc_auc['macro']),
                     color = 'navy', linestyle = ':', linewidth = 4)
            colors = cycle(['#0173b2', '#de8f05', '#029e73'])
            for i, color in zip(range(n_labels), colors):
                plt.plot(fpr[i], tpr[i], color = color, lw = 2,
                         label = 'ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-Class ROCs')
            plt.legend(bbox_to_anchor = (1, 1))
            plt.show()
        else:
            if cm_type == 'train':
                y_predicted = model.predict_proba(Xtrain)[:, 1]
                fpr, tpr, thres = roc_curve(ytrain, y_predicted)
                auc_val = roc_auc_score(ytrain, y_predicted)
            else:
                y_predicted = model.predict_proba(Xtest)[:, 1]
                fpr, tpr, thres = roc_curve(ytest, y_predicted)
                auc_val = roc_auc_score(ytest, y_predicted)

            def plot_roc_curve(fpr, tpr, label = None):
                plt.plot(fpr, tpr, linewidth=2, label = label)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.axis([0, 1, 0, 1])
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
            plot_roc_curve(fpr, tpr, 'Classifier')
            plt.legend(bbox_to_anchor = (1, 1))
            plt.title('ROC curve (AUC = %0.3f)' % auc_val)
            plt.show()


def cm_performance(cm,
                   classes):
    '''
    This function prints some performance values of a confusion matrix.
    CAREFUL: This function only works for 2 or 3 outcome categories.
    '''

    # check if number of labels is fine
    n_labels = len(cm)
    label_range = [2, 3]
    if n_labels not in label_range:
        raise ValueError('Invalid number of labels. Expected one of: %s' % label_range)

    # print label order
    print('Label Order:', classes)

    # confusion matrix:
    cm_eval = cm

    # Accuracy
    acc = np.diagonal(cm_eval).sum() / cm.sum()
    # Sensitivities
    sens = np.diagonal(cm_eval) / cm_eval.sum(axis=1)  # axis 1 = horizontal
    # Positive Predictive Values
    ppv = np.diagonal(cm_eval) / cm_eval.sum(axis=0)  # axis 0 = vertical
    # F1 scores
    f1 = 2 * (sens * ppv) / (sens + ppv)
    # Balanced Accuracy
    bac = sens.mean() # same as sklearn formula

    if n_labels == 3:
        # Specificities
        spec = [None] * 3
        spec[0] = (cm_eval[1, 1] + cm_eval[1, 2] + cm_eval[2, 1] + cm_eval[2, 2]) / (cm_eval[1, ].sum() + cm_eval[2, ].sum())
        spec[1] = (cm_eval[0, 0] + cm_eval[0, 2] + cm_eval[2, 0] + cm_eval[2, 2]) / (cm_eval[0, ].sum() + cm_eval[2, ].sum())
        spec[2] = (cm_eval[0, 0] + cm_eval[0, 1] + cm_eval[1, 0] + cm_eval[1, 1]) / (cm_eval[0, ].sum() + cm_eval[1, ].sum())
        # Negative Predictive Values
        npv = [None] * 3
        npv[0] = (cm_eval[1, 1] + cm_eval[1, 2] + cm_eval[2, 1] + cm_eval[2, 2]) / (cm_eval[:, 1].sum() + cm_eval[:, 2].sum())
        npv[1] = (cm_eval[0, 0] + cm_eval[0, 2] + cm_eval[2, 0] + cm_eval[2, 2]) / (cm_eval[:, 0].sum() + cm_eval[:, 2].sum())
        npv[2] = (cm_eval[0, 0] + cm_eval[0, 1] + cm_eval[1, 0] + cm_eval[1, 1]) / (cm_eval[:, 0].sum() + cm_eval[:, 1].sum())
    else:
        # Specificities
        spec = [None] * 2
        spec[0] = cm_eval[1, 1] / (cm_eval[1, ].sum())
        spec[1] = cm_eval[0, 0] / (cm_eval[0, ].sum())
        # Negative Predictive Values
        npv = [None] * 2
        npv[0] = cm_eval[1, 1] / (cm_eval[:, 1].sum())
        npv[1] = cm_eval[0, 0] / (cm_eval[:, 0].sum())

    # AUC
    auc = (np.array(spec) / 2) + (np.array(sens) / 2)

    # print everything
    print('Accuracy overall: {:.3f}'.format(acc))
    print('Balanced accuracy: {:.3f}'.format(bac))
    print('F1 scores for each group:', np.round(f1, 3))
    print('Sensitivity values for each group:', np.round(sens, 3))
    print('Specificity values for each group:', np.round(spec, 3))
    print('Positive Predictive Values for each group:', np.round(ppv, 3))
    print('Negative Predictive Values for each group:', np.round(npv, 3))
    print('AUC values for each group: ', np.round(auc, 3), 'WARNING: ROC with only one decision threshold significantly underestimates the true AUC!')
