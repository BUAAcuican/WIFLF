#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
==============================================================
ClassificationModel
==============================================================
The code is about two functions:
Selection_Classifications():

Build_Evaluation_Classification_Model():
.

@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com

==============================================================
"""
# print(__doc__)

import pandas as pd
import numpy as np
import math
import heapq
import random
import copy
import time
import os
from collections import Counter
from sklearn import preprocessing, model_selection, metrics,\
    linear_model, naive_bayes, tree, neighbors, ensemble, svm, neural_network
# MinMaxScaler, StandardScaler, Binarizer, Imputer, LabelBinarizer, FunctionTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from Performance import performance

def Selection_Classifications(clf_index, r):
    '''

    :param clf_index:
    :param r:
    :return:
    '''

    # dt = tree.DecisionTreeClassifier(random_state=r + 1)  # CART
    rf = ensemble.RandomForestClassifier(random_state=r + 1, n_estimators=100)
    # svml = svm.LinearSVC(random_state=r+1)
    svml = svm.SVC(kernel='linear', probability=True, random_state=r + 1,
               gamma='auto')  # SVM + Linear  Kernel (SVML)
    svmr = svm.SVC(kernel='rbf', probability=True, random_state=r + 1, gamma='auto')  # SVM + RBF Kernel (SVMR)
    svmp = svm.SVC(kernel='poly', probability=True, random_state=r + 1,
               gamma='auto')  # SVM+ Polynomial Kernel (SVMP)
    # Nusvm = svm.NuSVC(probability=True, random_state=r)
    mlp = neural_network.MLPClassifier(random_state=r + 1)
    knn = neighbors.KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='uniform')  # IB1 = knn IB1（KNN,K=1）
    gnb = naive_bayes.GaussianNB(priors=None)
    lr = linear_model.LogisticRegression(random_state=r + 1, solver='liblinear', max_iter=100, multi_class='ovr')

    clfs = [{'NB': gnb}, {'LR': lr}, {'RF': rf}, {'KNN': knn},
            {'SVMR': svmr}, {'MLP': mlp}, {'SVML': svml}, {'SVMP': svmp}]  #  {'DT': dt},

    classifier = clfs[clf_index]
    modelname = str(list(classifier.keys())[0])  # 分类器的名称
    clf = list(classifier.values())[0]  # 分类器参数模型
    return modelname, clf

def Build_Evaluation_Classification_Model(classifier, X_train, y_train, X_test, y_test):
    '''

    :param classifier:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return: measures: type: dict
    '''
    # print("X_train, y_train", X_train, y_train, type(X_train), type(y_train))
    # print(classifier, type(classifier))
    classifier.fit(X_train, y_train)
    # y_te_prepredict = classifier.predict(X_test)  # predict labels
    y_te_predict_proba = classifier.predict_proba(X_test)  # predict probability
    prob_pos = []  # the probability of predicting as positive
    prob_neg = list()  # the probability of predicting as negative
    # print(X_train, X_train.shape, y_train, y_train.shape,
    #       X_test, X_test.shape, y_test, y_test.shape, classifier.classes_)
    for j in range(y_te_predict_proba.shape[0]):  # 每个类的概率, array类型
        if len(classifier.classes_) <= 1:
            print('第%d个样本预测出错' % j)
        else:
            if classifier.classes_[1] == 1:
                prob_pos.append(y_te_predict_proba[j][1])
                prob_neg.append(y_te_predict_proba[j][0])
            else:
                prob_pos.append(y_te_predict_proba[j][0])
                prob_neg.append(y_te_predict_proba[j][1])
    prob_pos = np.array(prob_pos)
    prob_neg = np.array(prob_neg)
    measures = performance(y_test, prob_pos)

    return measures

if __name__ == '__main__':

   pass