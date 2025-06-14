import copy
import numpy as np
import pandas as pd

from scipy.special import softmax
from collections import defaultdict

from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, auc, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score


def get_PD_result(df, is_cls):

    PD2predList = defaultdict(list)
    PD2probList = defaultdict(list)
    PD2GT       = {}

    if (is_cls):
        for i in range(len(df)):
            PD, imgName_idx, gt_class, pred_class, pred_prob = df.iloc[i]
            pred_prob = str(pred_prob).replace('[', '').replace(']', '').strip().split()
            pred_prob = np.asarray(pred_prob, np.float64)
            pred_prob = softmax(pred_prob)

            PD2predList[PD].append(pred_class)
            PD2probList[PD].append(pred_prob[1])
            PD2GT[PD] = gt_class

        gt_classes   = []
        pred_classes = []
        pred_probs   = []

        for PD in PD2GT:
            preds = np.asarray(PD2predList[PD]).tolist()

            pos = np.sum(preds)
            neg = len(preds) - pos

            if (pos >= neg):
                pred = 1
            else:
                pred = 0

            probs = []
            for idx, prob in enumerate(PD2probList[PD]):
                if (preds[idx] == pred):
                    probs.append(prob)

            pred_classes.append(pred)
            pred_probs.append(np.mean(probs))
            gt_classes.append(PD2GT[PD])

        return gt_classes, pred_classes, pred_probs
    
    else:
        for i in range(len(df)):
            PD, imgName_idx, gt_value, pred_value = df.iloc[i]

            PD2predList[PD].append(pred_value)
            PD2GT[PD] = gt_value

        gt_values   = []
        pred_values = []

        for PD in PD2GT:
            preds = np.asarray(PD2predList[PD]).tolist()

            pred_values.append(np.mean(preds))
            gt_values.append(PD2GT[PD])

            # print(f'{PD2GT[PD]:.2f} {np.mean(preds):.2f} {np.round(preds, 2)}')

        return gt_values, pred_values


def cal_score(gt, pred, pred_prob=None, thr=None, is_cls=None):

    assert type(gt)   == list
    assert type(pred) == list

    if (is_cls):
        pred_class = copy.deepcopy(pred)
        gt_class   = copy.deepcopy(gt)
    else:
        pred_class = (np.asarray(copy.deepcopy(pred)) >= thr).astype(np.int16).tolist()
        gt_class   = (np.asarray(copy.deepcopy(gt))   >= thr).astype(np.int16).tolist()

    cfm = confusion_matrix(gt_class, pred_class)
    tn, fp, fn, tp = cfm.ravel()
    cfm = [tp, tn, fp, fn]

    if (is_cls):
        # auroc     = roc_auc_score(gt_class, pred_prob)
        auroc     = 1
        prc       = precision_recall_curve(gt_class, pred_prob)
        precision = prc[0]
        recall    = prc[1]
        threshold = prc[2]
        auprc     = auc(recall, precision)
    else:
        try:
            r2, p_value = pearsonr(gt, pred)
            mae = sklearn.metrics.mean_absolute_error(gt, pred)
        except:
            r2, p_value = 0, 0
            mae = 10000000


    accuracy      = accuracy_score(gt_class, pred_class)
    BA            = balanced_accuracy_score(gt_class, pred_class)
    recall        = recall_score(gt_class, pred_class, zero_division=0)
    specificity   = recall_score(gt_class, pred_class, pos_label=0, zero_division=0)
    PPV           = precision_score(gt_class, pred_class, zero_division=0)
    NPV           = precision_score(gt_class, pred_class, pos_label=0, zero_division=0)
    f1            = f1_score(gt_class, pred_class, zero_division=0)

    if (is_cls):
        return [cfm, accuracy, BA, recall, specificity, PPV, NPV, f1, auroc, auprc]
    else:
        return [cfm, accuracy, BA, recall, specificity, PPV, NPV, f1, r2, p_value, mae]
