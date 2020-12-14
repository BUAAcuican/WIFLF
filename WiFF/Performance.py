#!/usr/bin/env python
# -*- coding: UTF-8 -*-


"""
==============================================================
Performance
==============================================================
The code is about Performance for Bianary classification.

@ author: cuican
@ email: cuican5100923@163.com
@
==============================================================
"""
# print(__doc__)

import pandas as pd
import numpy as np
import time
from sklearn import metrics

def performance(y_true, prob_pos):
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # sklearn.metrics.confusion_matrix
    :param y_true:  A 1-D array that denotes test instances' actual labels, {0,1} where 1 is defective, 0 non-defective.
    :param prob_pos: A 1-D array, which denotes the probability of predicted being defective for each instance.
    :return: A dict including multiple elements, such as PD, PF, AUC,..., MCC.
    """
    TN = 0  # initialize the global variables
    FP = 0
    FN = 0
    TP = 0
    confMat = np.zeros((2, 2))  # initialize the global variables
    y_pred = (prob_pos >= 0.5).astype(np.int32)
    alpha = 4  # skewed_F_measure中的取值
    PD = 0
    PF = 0  # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    Bal = 0
    G_Measure = 0
    G_Mean2 = 0
    MCC = 0
    Pre = 0
    F1 = 0
    skewed_F_measure = 0
    AUC = 0

    if np.sum(y_true) == 0:
        # confMat = metrics.confusion_matrix(y_true, y_pred)  # confMat = [[len(y_true)]], shape: (1,1)
        # TN, FP, FN, TP = confMat.ravel()
        TN = len(y_pred) - np.sum(y_pred)
        FP = np.sum(y_pred)
        FN = 0
        TP = 0
        confMat = np.array([[TN, FP], [FN, TP]])
        # raise ValueError('y_trues are all non-defective, please check...')
    elif np.sum(y_true) == len(y_true):
        # confMat = metrics.confusion_matrix(y_true, y_pred)   # confMat = [[len(y_true)]], shape: (1,1)
        # TN, FP, FN, TP = confMat.ravel()
        TN = 0
        FP = 0
        FN = len(y_pred) - np.sum(y_pred)
        TP = np.sum(y_pred)
        confMat = np.array([[TN, FP], [FN, TP]])
        # raise ValueError('y_trues are all defective, please check...')
    else:
        confMat = metrics.confusion_matrix(y_true, y_pred)  # [[TN, FP],[FN, TP]], i.e., C00 is TN, C11 is TP.
        # TN = confMat[0, 0]
        # FP = confMat[0, 1]
        # FN = confMat[1, 0]
        # TP = confMat[1, 1]
        TN, FP, FN, TP = confMat.ravel()

    # AUC = metrics.roc_auc_score(y_true, prob_pos)
    # print(AUC)
    # print('confMat:', confMat)
    # print('TN, FP, FN, TP:', TN, FP, FN, TP)

    # Performance
    Acc = metrics.accuracy_score(y_true, y_pred)
    Error = (FP + FN) / (TP + TN + FP + FN)
    # Case1：存在正样本，负样本不一定存在
    if TN == 0 and TP != 0:
        # raise ValueError('negative samples are all divided wrongly, please check...')
        if FN == 0 and FP == 0:  # 负样本没分对也没分错，正样本全分错了，只存在+样本
            raise ValueError('all samples are positive, please check...')
        elif FN != 0 and FP == 0:  # 负样本没分对也没分错，只存在+样本
            raise ValueError('all samples are positive, please check...')
        elif FN == 0 and FP != 0:  # 负样本全分错了，正样本全分对了,存在+-样本
            MCC = np.nan  # 分母为0
            Pre = metrics.precision_score(y_true, y_pred)
            F1 = metrics.f1_score(y_true, y_pred)
            PD = metrics.recall_score(y_true, y_pred)  # Recall = TP / (TP + FN)
            PF = FP / (FP + TN)  # FPR, PF
            Bal = 1 - (np.sqrt((np.power(0 - PF, 2) + np.power(1 - PD, 2)) / 2))
            G_Measure = 2 * PD * (1 - PF) / (PD + (1 - PF))
            G_Mean2 = np.sqrt(PD * (1 - PF))  # G_Mean2
            skewed_F_measure = ((1 + alpha) * Pre * PD) / (alpha * Pre + PD)
            AUC = metrics.roc_auc_score(y_true, prob_pos)
            TNR = TN / (TN + FP)  # Specificity
            FNR = FN / (FN + TP)
        else:  # FN != 0 and FP != 0 负样本全分错了，正样本有错有对，存在+-样本
            MCC = metrics.matthews_corrcoef(y_true, y_pred)
            Pre = metrics.precision_score(y_true, y_pred)
            F1 = metrics.f1_score(y_true, y_pred)
            PD = metrics.recall_score(y_true, y_pred)  # Recall = TP / (TP + FN)
            PF = FP / (FP + TN)  # FPR, PF
            Bal = 1 - (np.sqrt((np.power(0 - PF, 2) + np.power(1 - PD, 2)) / 2))
            G_Measure = 2 * PD * (1 - PF) / (PD + (1 - PF))
            G_Mean2 = np.sqrt(PD * (1 - PF))  # G_Mean2
            skewed_F_measure = ((1 + alpha) * Pre * PD) / (alpha * Pre + PD)
            AUC = metrics.roc_auc_score(y_true, prob_pos)
            TNR = TN / (TN + FP)  # Specificity
            FNR = FN / (FN + TP)
    # Case2：存在负样本，+样本不一定存在
    elif TN != 0 and TP == 0:
        if FP != 0 and FN != 0:  # 负样本分和正样本都分错了，且正样本全错了
            F1 = np.nan  # 分母为0
            skewed_F_measure = np.nan  # 分母为0
            AUC = metrics.roc_auc_score(y_true, prob_pos)
            MCC = metrics.matthews_corrcoef(y_true, y_pred)
            Pre = metrics.precision_score(y_true, y_pred)
            PD = metrics.recall_score(y_true, y_pred)  # Recall = TP / (TP + FN)
            PF = FP / (FP + TN)  # FPR, PF
            Bal = 1 - (np.sqrt((np.power(0 - PF, 2) + np.power(1 - PD, 2)) / 2))
            G_Measure = 2 * PD * (1 - PF) / (PD + (1 - PF))
            G_Mean2 = np.sqrt(PD * (1 - PF))  # G_Mean2
            TNR = TN / (TN + FP)  # Specificity
            FNR = FN / (FN + TP)
        elif FP == 0 and FN != 0:  # 正样本全分错了，负样本全分对了
            F1 = np.nan  # 分母为0
            skewed_F_measure = np.nan  # 分母为0
            AUC = metrics.roc_auc_score(y_true, prob_pos)
            MCC = np.nan  # 分母为0
            Pre = np.nan  # 分母为0
            PD = metrics.recall_score(y_true, y_pred)  # Recall = TP / (TP + FN)
            PF = FP / (FP + TN)  # FPR, PF
            Bal = 1 - (np.sqrt((np.power(0 - PF, 2) + np.power(1 - PD, 2)) / 2))
            G_Measure = 2 * PD * (1 - PF) / (PD + (1 - PF))
            G_Mean2 = np.sqrt(PD * (1 - PF))  # G_Mean2
            TNR = TN / (TN + FP)  # Specificity
            FNR = FN / (FN + TP)
        elif FP == 0 and FN == 0:  # 正样本没分对也没分错，负样本没分错,只有负类
            raise ValueError('all samples are negative, please check...')
        else:  # FP != 0 and FN == 0 正样本没分对没分错，负样本分错了，只有负类
            raise ValueError('all samples are negative, please check...')
    # Case3: # 负样本和正样本全分错了，两种样本是否存在看情况
    elif TN == 0 and TP == 0:
        if FP != 0 and FN != 0:  # 负样本分和正样本全都分错了，且+-样本都存在
            F1 = np.nan  # 分母为0
            skewed_F_measure = np.nan  # 分母为0
            AUC = metrics.roc_auc_score(y_true, prob_pos)
            MCC = metrics.matthews_corrcoef(y_true, y_pred)
            Pre = metrics.precision_score(y_true, y_pred)
            PD = metrics.recall_score(y_true, y_pred)  # Recall = TP / (TP + FN)
            PF = FP / (FP + TN)  # FPR, PF
            Bal = 1 - (np.sqrt((np.power(0 - PF, 2) + np.power(1 - PD, 2)) / 2))
            G_Measure = 2 * np.nan  # 分母为0
            G_Mean2 = np.sqrt(PD * (1 - PF))  # G_Mean2
            TNR = TN / (TN + FP)  # Specificity
            FNR = FN / (FN + TP)
        elif FP == 0 and FN != 0:  # 正样本全分错了，负样本没分对也没分错,只有+类
            raise ValueError('all samples are Positive, please check...')
        elif FP == 0 and FN == 0:  # 正样本没分对也没分错，负样本没分对没分错,没有+-样本
            raise ValueError('None samples, please check...')
        else:  # FP != 0 and FN == 0 正样本没分对没分错，负样本分错了，只有-类
            raise ValueError('all samples are negative, please check...')
    # case4: TP != 0 and TN != 0
    else:
        Pre = metrics.precision_score(y_true, y_pred)
        F1 = metrics.f1_score(y_true, y_pred)
        PD = metrics.recall_score(y_true, y_pred)  # Recall = TP / (TP + FN)
        PF = FP / (FP + TN)  # FPR, PF
        Bal = 1 - (np.sqrt((np.power(0 - PF, 2) + np.power(1 - PD, 2)) / 2))
        G_Measure = 2 * PD * (1 - PF) / (PD + (1 - PF))
        G_Mean2 = np.sqrt(PD * (1 - PF))  # G_Mean2
        skewed_F_measure = ((1 + alpha) * Pre * PD) / (alpha * Pre + PD)
        AUC = metrics.roc_auc_score(y_true, prob_pos)
        MCC = metrics.matthews_corrcoef(y_true, y_pred)
        TNR = TN / (TN + FP)  # Specificity
        FNR = FN / (FN + TP)

    # 'PD(Recall, TPR)': PD, 'PF(FPR)': PF,
    return {'AUC': AUC, 'Balance': Bal, 'MCC': MCC, 'G-Measure': G_Measure, 'G-Mean': G_Mean2,
            'PD': PD, 'PF': PF, 'skewed_F': skewed_F_measure, 'F1': F1,
            'Accuracy': Acc, 'Error': Error, 'Precision': Pre, 'TNR': TNR, 'FNR': FNR,
            'TP':  TP, 'TN': TN, 'FP': FP, 'FN': FN}

if __name__ == '__main__':
    y_true1 = [1, 0, 1, 0]
    y_true2 = [0, 0, 0, 0]
    y_true3 = [1, 1, 1, 1]
    y_true5 = [0, 1, 0, 1, 0]
    prob_pos1 = np.array([1, 0, 0, 1])
    prob_pos2 = np.array([0, 0, 0, 0])
    prob_pos3 = np.array([1, 1, 1, 1])
    prob_pos4 = np.array([0, 0, 0, 1])

    prob_pos5 = np.array([0, 0, 0, 0, 0])
    prob_pos6 = np.array([1, 0, 1, 0, 1])
    prob_pos7 = np.array([1, 1, 1, 1, 1])
    # performance(y_true1, prob_pos1)
    # performance(y_true1, prob_pos2)
    # performance(y_true1, prob_pos3)
    # performance(y_true1, prob_pos4)  # 仅TP=0
    # performance(y_true5, prob_pos5)  # 仅TP=0,FP=0
    # performance(y_true5, prob_pos6)  # 仅TP=0,TN=0
    performance(y_true5, prob_pos7)  # 仅FN=0,TN=0

    # performance(y_true2, prob_pos1)  # SKEWD-F1 =nan, AUC error
    # performance(y_true2, prob_pos2)  # SKEWD-F1 and  AUC error
    # performance(y_true2, prob_pos3)  # SKEWD-F1 =nan, AUC error


    # performance(y_true3, prob_pos1)
    # performance(y_true3, prob_pos2)
    # performance(y_true3, prob_pos3)
