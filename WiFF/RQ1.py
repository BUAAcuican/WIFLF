#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
Answering RQ1 for Paper: How does WiFF work?

    CPDP_pre: preprocessing
    CPDP_SMOTE: preprocessing + SMOTE
    CPDP_iF: preprocessing + iForest
    CPDP_iFl: preprocessing + iForest + class information
    CPDP_SMOTE_iF: preprocessing + SMOTE + iForest
    CPDP_SMOTE_iFl_d/b: preprocessing + SMOTE + iForest + class information (default/best parameters)
    WiFF: preprocessing + SMOTE + WiFF

Weighted Isolation Forest Filter (Combined with SMOTE)
for Instance-based Cross-Project Defect Prediction
============================================================================
@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
@2020/10/20
============================================================================
"""
# print(__doc__)
from __future__ import print_function

# print(__doc__)

import pandas as pd
import numpy as np
import math
import heapq
import random
import copy
import time
import os

from WiFF import WiFF_main, IForest
from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model
from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Standard_Features

import multiprocessing
from functools import partial
from Statistical_Test import WinTieLoss, Cohens_d, Wilcoxon_signed_rank_test

def CPDP_SMOTE(mode, clf_index, runtimes):

    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # datasets for RQ1
    datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.4.csv', 'xerces-init.csv']]

    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv']]

    datanum = 0
    for i in range(len(datasets)):
        datanum = datanum + len(datasets[i])
    # print(datanum)
    # mode = [preprocess_mode, train_mode, save_file_name]
    # preprocess_mode = mode[0]
    iForest_parameters = mode[0]
    train_mode = mode[1]
    save_file_name = mode[2]

    df_file_measures = pd.DataFrame()  # the measures of all files in all runtimes
    classifiername = []
    # file_list = os.listdir(fpath)

    n = 0
    for i in range(len(datasets)):
        for file_te in datasets[i]:
            n = n + 1
            print('----------%s:%d/%d------' % ('Dataset', n, datanum))
            # print('testfile', file_te)
            start_time = time.time()

            Address_te = datapath + file_te
            Samples_te = NumericStringLabel2BinaryLabel(Address_te)  # DataFrame
            data = Samples_te.values  # DataFrame2Array
            X = data[:, : -1]  # test features
            y = data[:, -1]  # test labels
            Sample_tr0 = NumericStringLabel2BinaryLabel(Address_te)
            column_name = Sample_tr0.columns.values  # the column name of the data

            df_r_measures = pd.DataFrame()  # the measures of each file in runtimes
            for r in range(runtimes):
                if train_mode == 'M2O_CPDP':  # train data contains more files different from the test data project
                    X_test = X
                    y_test = y
                    Samples_tr_all = pd.DataFrame()  # initialize the candidate training data of all cp data
                    for file_tr in datasets[i]:
                        if file_tr != file_te:
                            # print('train_file:', file_tr)
                            Address_tr = datapath + file_tr
                            Samples_tr = NumericStringLabel2BinaryLabel(Address_tr)  # original train data, DataFrame
                            Samples_tr.columns = column_name.tolist()  # 批量更改列名
                            Samples_tr_all = pd.concat([Samples_tr_all, Samples_tr], ignore_index=False, axis=0,
                                                       sort=False)
                    # Samples_tr_all.to_csv(f2, index=None, columns=None)  # 将类标签二元化的数据保存，保留列名，不增加行索引

                    # /* step1: data preprocessing */
                    Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug', r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    target = np.c_[X_test_zscore, y_test]

                    # /* step2: pass */

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_cpdp_smote, model_cpdp_smote = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_cpdp_smote)
                    # print("modelname_cpdp_smote:", modelname_cpdp_smote)
                    measures_cpdp_smote = Build_Evaluation_Classification_Model(model_cpdp_smote, X_train_zscore, y_train, X_test_zscore, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_cpdp_smote.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_zscore), 'test_len': len(X_test_zscore),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_cpdp_smote,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_cpdp_smote, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False, ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures


def CPDP_iF(mode, clf_index, runtimes):

    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # datasets for RQ1
    datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.4.csv', 'xerces-init.csv']]

    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv']]

    datanum = 0
    for i in range(len(datasets)):
        datanum = datanum + len(datasets[i])
    # print(datanum)
    # mode = [preprocess_mode, train_mode, save_file_name]
    # preprocess_mode = mode[0]
    iForest_parameters = mode[0]
    train_mode = mode[1]
    save_file_name = mode[2]

    df_file_measures = pd.DataFrame()  # the measures of all files in all runtimes
    classifiername = []
    # file_list = os.listdir(fpath)

    n = 0
    for i in range(len(datasets)):
        for file_te in datasets[i]:
            n = n + 1
            print('----------%s:%d/%d------' % ('Dataset', n, datanum))
            # print('testfile', file_te)
            start_time = time.time()

            Address_te = datapath + file_te
            Samples_te = NumericStringLabel2BinaryLabel(Address_te)  # DataFrame
            data = Samples_te.values  # DataFrame2Array
            X = data[:, : -1]  # test features
            y = data[:, -1]  # test labels
            Sample_tr0 = NumericStringLabel2BinaryLabel(Address_te)
            column_name = Sample_tr0.columns.values  # the column name of the data

            df_r_measures = pd.DataFrame()  # the measures of each file in runtimes
            for r in range(runtimes):
                if train_mode == 'M2O_CPDP':  # train data contains more files different from the test data project
                    X_test = X
                    y_test = y
                    Samples_tr_all = pd.DataFrame()  # initialize the candidate training data of all cp data
                    for file_tr in datasets[i]:
                        if file_tr != file_te:
                            # print('train_file:', file_tr)
                            Address_tr = datapath + file_tr
                            Samples_tr = NumericStringLabel2BinaryLabel(Address_tr)  # original train data, DataFrame
                            Samples_tr.columns = column_name.tolist()  # 批量更改列名
                            Samples_tr_all = pd.concat([Samples_tr_all, Samples_tr], ignore_index=False, axis=0,
                                                       sort=False)
                    # Samples_tr_all.to_csv(f2, index=None, columns=None)  # 将类标签二元化的数据保存，保留列名，不增加行索引

                    # /* step1: data preprocessing */
                    Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug', r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    target = np.c_[X_test_zscore, y_test]

                    # /* step2: iFF(iForest Filter) */
                    # *******************WiFF*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))

                    # iForest_parameters = [itree_num, subsamples, hlim, mode, s0, alpha]
                    itree_num = iForest_parameters[0]
                    subsamples = iForest_parameters[1]
                    hlim = iForest_parameters[2]
                    mode_if = iForest_parameters[3]
                    s0 = iForest_parameters[4]
                    alpha = iForest_parameters[5]

                    forest = IForest(itree_num, subsamples, hlim)  # build an iForest(t, subsample, hlim)
                    s, rate, abnormal_list = forest.Build_Forest(X_train, mode_if, r, s0, alpha)
                    ranges_X_train, Sample_X_train = Delete_abnormal_samples(X_train, abnormal_list)
                    ranges_y_train, Sample_y_train = Delete_abnormal_samples(y_train, abnormal_list)

                    X_train_new = Sample_X_train
                    y_train_new = Sample_y_train # np.array(Sample_y_train.tolist() + Sample_y_train_pos_weight.tolist())  # list垂直拼接

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_cpdp_if, model_cpdp_if = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_cpdp_if)
                    # print("modelname_cpdp_if:", modelname_cpdp_if)
                    measures_cpdp_if = Build_Evaluation_Classification_Model(model_cpdp_if, X_train_new, y_train_new, X_test_zscore, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_cpdp_if.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_cpdp_if,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_cpdp_if, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False, ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures

def CPDP_iFl(mode, clf_index, runtimes):

    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # datasets for RQ1
    datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.4.csv', 'xerces-init.csv']]

    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv']]

    datanum = 0
    for i in range(len(datasets)):
        datanum = datanum + len(datasets[i])
    # print(datanum)
    # mode = [preprocess_mode, train_mode, save_file_name]
    # preprocess_mode = mode[0]
    iForest_parameters = mode[0]
    train_mode = mode[1]
    save_file_name = mode[2]

    df_file_measures = pd.DataFrame()  # the measures of all files in all runtimes
    classifiername = []
    # file_list = os.listdir(fpath)

    n = 0
    for i in range(len(datasets)):
        for file_te in datasets[i]:
            n = n + 1
            print('----------%s:%d/%d------' % ('Dataset', n, datanum))
            # print('testfile', file_te)
            start_time = time.time()

            Address_te = datapath + file_te
            Samples_te = NumericStringLabel2BinaryLabel(Address_te)  # DataFrame
            data = Samples_te.values  # DataFrame2Array
            X = data[:, : -1]  # test features
            y = data[:, -1]  # test labels
            Sample_tr0 = NumericStringLabel2BinaryLabel(Address_te)
            column_name = Sample_tr0.columns.values  # the column name of the data

            df_r_measures = pd.DataFrame()  # the measures of each file in runtimes
            for r in range(runtimes):
                if train_mode == 'M2O_CPDP':  # train data contains more files different from the test data project
                    X_test = X
                    y_test = y
                    Samples_tr_all = pd.DataFrame()  # initialize the candidate training data of all cp data
                    for file_tr in datasets[i]:
                        if file_tr != file_te:
                            # print('train_file:', file_tr)
                            Address_tr = datapath + file_tr
                            Samples_tr = NumericStringLabel2BinaryLabel(Address_tr)  # original train data, DataFrame
                            Samples_tr.columns = column_name.tolist()  # 批量更改列名
                            Samples_tr_all = pd.concat([Samples_tr_all, Samples_tr], ignore_index=False, axis=0,
                                                       sort=False)
                    # Samples_tr_all.to_csv(f2, index=None, columns=None)  # 将类标签二元化的数据保存，保留列名，不增加行索引

                    # /* step1: data preprocessing */
                    Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug', r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    target = np.c_[X_test_zscore, y_test]

                    # /* step2: iFF+label(iForest Filter+label) */
                    # *******************WiFF*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train_neg0, X_train_pos0, y_train_neg0, y_train_pos0 = DevideData2TwoClasses(X_train_zscore, y_train)  # original X+,y+,X-,y-

                    # iForest_parameters = [itree_num, subsamples, hlim, mode, s0, alpha]
                    itree_num = iForest_parameters[0]
                    subsamples = iForest_parameters[1]
                    hlim = iForest_parameters[2]
                    mode_if = iForest_parameters[3]
                    s0 = iForest_parameters[4]
                    alpha = iForest_parameters[5]

                    forest = IForest(itree_num, subsamples, hlim)  # build an iForest(t, subsample, hlim)
                    s, rate, abnormal_list_neg = forest.Build_Forest(X_train_neg0, mode_if, r, s0, alpha)
                    ranges_X_train_neg, Sample_X_train_neg = Delete_abnormal_samples(X_train_neg0, abnormal_list_neg)
                    ranges_y_train_neg, Sample_y_train_neg = Delete_abnormal_samples(y_train_neg0, abnormal_list_neg)

                    s_weight, rate_weight, abnormal_list_pos = forest.Build_Forest(X_train_pos0, mode_if, r, s0, alpha)
                    ranges_X_train_pos, Sample_X_train_pos = Delete_abnormal_samples(X_train_pos0, abnormal_list_pos)
                    ranges_y_train_pos, Sample_y_train_pos = Delete_abnormal_samples(y_train_pos0, abnormal_list_pos)
                    X_train_new = np.concatenate((Sample_X_train_neg, Sample_X_train_pos), axis=0)  # array垂直拼接
                    y_train_new = np.array(Sample_y_train_neg.tolist() + Sample_y_train_pos.tolist())  # list垂直拼接

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_cpdp_ifl, model_cpdp_ifl = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_cpdp_ifl)
                    # print("modelname_cpdp_ifl:", modelname_cpdp_ifl)
                    measures_cpdp_ifl = Build_Evaluation_Classification_Model(model_cpdp_ifl, X_train_new, y_train_new, X_test_zscore, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_cpdp_ifl.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_cpdp_ifl,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_cpdp_ifl, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False, ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures

def CPDP_SMOTE_iF(mode, clf_index, runtimes):

    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # datasets for RQ1
    datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.4.csv', 'xerces-init.csv']]

    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv']]

    datanum = 0
    for i in range(len(datasets)):
        datanum = datanum + len(datasets[i])
    # print(datanum)
    # mode = [preprocess_mode, train_mode, save_file_name]
    # preprocess_mode = mode[0]
    iForest_parameters = mode[0]
    train_mode = mode[1]
    save_file_name = mode[2]

    df_file_measures = pd.DataFrame()  # the measures of all files in all runtimes
    classifiername = []
    # file_list = os.listdir(fpath)

    n = 0
    for i in range(len(datasets)):
        for file_te in datasets[i]:
            n = n + 1
            print('----------%s:%d/%d------' % ('Dataset', n, datanum))
            # print('testfile', file_te)
            start_time = time.time()

            Address_te = datapath + file_te
            Samples_te = NumericStringLabel2BinaryLabel(Address_te)  # DataFrame
            data = Samples_te.values  # DataFrame2Array
            X = data[:, : -1]  # test features
            y = data[:, -1]  # test labels
            Sample_tr0 = NumericStringLabel2BinaryLabel(Address_te)
            column_name = Sample_tr0.columns.values  # the column name of the data

            df_r_measures = pd.DataFrame()  # the measures of each file in runtimes
            for r in range(runtimes):
                if train_mode == 'M2O_CPDP':  # train data contains more files different from the test data project
                    X_test = X
                    y_test = y
                    Samples_tr_all = pd.DataFrame()  # initialize the candidate training data of all cp data
                    for file_tr in datasets[i]:
                        if file_tr != file_te:
                            # print('train_file:', file_tr)
                            Address_tr = datapath + file_tr
                            Samples_tr = NumericStringLabel2BinaryLabel(Address_tr)  # original train data, DataFrame
                            Samples_tr.columns = column_name.tolist()  # 批量更改列名
                            Samples_tr_all = pd.concat([Samples_tr_all, Samples_tr], ignore_index=False, axis=0,
                                                       sort=False)
                    # Samples_tr_all.to_csv(f2, index=None, columns=None)  # 将类标签二元化的数据保存，保留列名，不增加行索引

                    # /* step1: data preprocessing */
                    Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug', r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    target = np.c_[X_test_zscore, y_test]

                    # /* step2: WiFF(weighted iForest Filter) */
                    # *******************WiFF*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train_resampled, y_train_resampled = DataImbalance(X_train_zscore, y_train, r, mode='smote')  # Data imbalance

                    # iForest_parameters = [itree_num, subsamples, hlim, mode, s0, alpha]
                    itree_num = iForest_parameters[0]
                    subsamples = iForest_parameters[1]
                    hlim = iForest_parameters[2]
                    mode_if = iForest_parameters[3]
                    s0 = iForest_parameters[4]
                    alpha = iForest_parameters[5]

                    forest = IForest(itree_num, subsamples, hlim)  # build an iForest(t, subsample, hlim)
                    s, rate, abnormal_list = forest.Build_Forest(X_train_resampled, mode_if, r, s0, alpha)
                    ranges_X_train, Sample_X_train = Delete_abnormal_samples(X_train_resampled, abnormal_list)
                    ranges_y_train, Sample_y_train = Delete_abnormal_samples(y_train_resampled, abnormal_list)
                    X_train_new = Sample_X_train
                    y_train_new = Sample_y_train
                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_cpdp_smote_if, model_cpdp_smote_if = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_cpdp_smote_if)
                    # print("modelname_cpdp_smote_if:", modelname_cpdp_smote_if)
                    measures_cpdp_smote_if = Build_Evaluation_Classification_Model(model_cpdp_smote_if, X_train_new, y_train_new, X_test_zscore, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_cpdp_smote_if.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_cpdp_smote_if,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_cpdp_smote_if, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False, ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures

def Main_WiFFd(runtimes):
    iForest_parameters = [100, 256, 255, 'rate', 0.5, 0.1]  # default values of iForest

    mode = [iForest_parameters, 'M2O_CPDP', 'WiFFd']
    for i in range(1, 3, 1):
        WiFF_main(mode=mode, clf_index=i, runtimes=runtimes)

def Main_CPDP_SMOTE(runtimes):

    mode = ["", 'M2O_CPDP', 'CPDP_SMOTE']
    for i in range(1, 3, 1):
        CPDP_SMOTE(mode=mode, clf_index=i, runtimes=runtimes)

def Main_CPDP_iF(runtimes):
    iForest_parameters = [100, 256, 255, 'rate', 0.5, 0.1]  # default values of iForest

    mode = [iForest_parameters, 'M2O_CPDP', 'CPDP_iF']
    for i in range(1, 3, 1):
        CPDP_iF(mode=mode, clf_index=i, runtimes=runtimes)

def Main_CPDP_iFl(runtimes):
    iForest_parameters = [100, 256, 255, 'rate', 0.5, 0.1]  # default values of iForest

    mode = [iForest_parameters, 'M2O_CPDP', 'CPDP_iFl']
    for i in range(1, 3, 1):
        CPDP_iFl(mode=mode, clf_index=i, runtimes=runtimes)

def Main_CPDP_SMOTE_iF(runtimes):
    iForest_parameters = [100, 256, 255, 'rate', 0.5, 0.1]  # default values of iForest

    mode = [iForest_parameters, 'M2O_CPDP', 'CPDP_SMOTE_iF']
    for i in range(1, 3, 1):
        CPDP_SMOTE_iF(mode=mode, clf_index=i, runtimes=runtimes)

def somefunc(iterable_term):
    '''
    :param iterable_term: pool id
    :return:
    '''
    times = 2
    if iterable_term == 1:
        return Main_CPDP_SMOTE(times)
    elif iterable_term == 5:
        return Main_CPDP_iF(times)
    elif iterable_term == 2:
        return Main_CPDP_iFl(times)
    elif iterable_term == 3:
        return Main_CPDP_SMOTE_iF(times)
    elif iterable_term == 4:
        return Main_WiFFd(times)
    else:
        pass




if __name__ == '__main__':
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    starttime = time.clock()  # 计时开始

    iterable = [1, 2, 3, 4, 5, 6, 7, 8]  #
    pool = multiprocessing.Pool(processes=5)
    func = partial(somefunc)
    pool.map(func, iterable)
    pool.close()
    pool.join()




    endtime = time.clock()  # 计时结束

    totaltime = endtime - starttime
    print('程序结束运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序共运行时间为:', totaltime)


