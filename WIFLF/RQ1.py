#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
Answering RQ1 for Paper: Does WIFLF work?

    NaiveMethod(NM): Naive method for M2O-CPDP, which only contains label transferring, data sampling and duplicate instances dropping of the data preprocessing step;
    ZscoreMethod(ZM): Zscore Method  for M2O-CPDP, and the method contains the data preprocessing step;
    ZscoreSMOTEMethod(ZSM): Zscore SMOTE Method for M2O-CPDP, and the approach includes data preprocessing and SMOTE;
    IsolationForestFilter(IFF): preprocessing + SMOTE + iForest
    WeightedIsolationForestFilter(WIFF): preprocessing + SMOTE + WIFLF
    WeightedIsolationForestwithLabelinformationFilter(WIFLF): preprocessing + SMOTE + WIFLF + class information

`Weighted Isolation Forest with Label Information Filter for Cross-Project Defect Prediction with Common Metrics`
============================================================================
@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
@2021/4/9
============================================================================
"""
# print(__doc__)
from __future__ import print_function

import pandas as pd
import numpy as np
import time
import os

import multiprocessing
from functools import partial

from isolationforest import detect_anomalies
from weightedisolationforest import detect_anomalies_improved
from weightedisolationforestpluslabel import detect_anomalies_improvedplusLabel
from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model

from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Standard_Features # Process_Data,

def NaiveMethod(mode, clf_index, runtimes):

    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    # # datasets for RQ1
    datasets = [['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv',
                'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv',
                'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv']]


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
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug',
                                                            r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]

                    # /* step2: naive*/
                    # *******************naive-CPDP*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_naive, model_naive = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_naive)
                    # print("modelname_wiff:", modelname_wiff)
                    measures_wiff = Build_Evaluation_Classification_Model(model_naive, X_train, y_train,
                                                                          X_test, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_wiff.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_naive,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_wiff, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False,
                                         ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures

def ZscoreMethod(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    # # datasets for RQ1
    datasets = [['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv',
                'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv',
                'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv']]

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
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug',
                                                            r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]

                    # /* step2: zscore*/
                    # *******************zscore-CPDP*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    # target = np.c_[X_test_zscore, y_test]
                    X_train_new = X_train_zscore
                    y_train_new = y_train

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_zscore, model_zscore = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_zscore)
                    # print("modelname_zscore:", modelname_zscore)
                    measures_wiff = Build_Evaluation_Classification_Model(model_zscore, X_train_new, y_train_new,
                                                                          X_test_zscore,
                                                                          y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_wiff.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_zscore,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_wiff, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False,
                                         ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures

def ZscoreSMOTEMethod(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    # # datasets for RQ1
    datasets = [['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv',
                 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv',
                 'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv']]

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
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug',
                                                            r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]

                    # /* step2: zscore+SMOTE*/
                    # *******************zscore-smote-CPDP*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    X_train_resampled, y_train_resampled = DataImbalance(X_train_zscore, y_train, r,
                                                                         mode='smote')  # Data imbalance
                    X_train_new = X_train_resampled
                    y_train_new = y_train_resampled

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_zs, model_zs = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_zs)
                    # print("modelname_zs:", modelname_zs)
                    measures_zs = Build_Evaluation_Classification_Model(model_zs, X_train_new, y_train_new,
                                                                          X_test_zscore,
                                                                          y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_zs.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_zs,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_zs, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False,
                                         ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures

def IsolationForestFilter(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    # # datasets for RQ1
    datasets = [['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv',
                 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv',
                 'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv']]

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
            print('testfile:', file_te)
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
                            print('train_file:', file_tr)
                            Address_tr = datapath + file_tr
                            Samples_tr = NumericStringLabel2BinaryLabel(Address_tr)  # original train data, DataFrame
                            Samples_tr.columns = column_name.tolist()  # 批量更改列名
                            Samples_tr_all = pd.concat([Samples_tr_all, Samples_tr], ignore_index=False, axis=0,
                                                       sort=False)
                    # Samples_tr_all.to_csv(f2, index=None, columns=None)  # 将类标签二元化的数据保存，保留列名，不增加行索引

                    # /* step1: data preprocessing */
                    Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug',
                                                            r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]

                    # /* step2: Zscore_SMOTE_iFF( Zscore +smote + iForest Filter) */
                    # ******************* Zscore_SMOTE_iFF*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    X_train_resampled, y_train_resampled = DataImbalance(X_train_zscore, y_train, r,
                                                                         mode='smote')  # Data imbalance

                    data_train_smote = pd.DataFrame(np.c_[X_train_resampled, y_train_resampled])
                    # iForest_parameters = [itree_num, subsamples, hlim, mode, s0, alpha]
                    itree_num = iForest_parameters[0]
                    subsamples = iForest_parameters[1]
                    hlim = iForest_parameters[2]
                    mode_if = iForest_parameters[3]
                    s0 = iForest_parameters[4]
                    alpha = iForest_parameters[5]
                    X_train_new_idx, X_train_new, y_train_new = detect_anomalies(data_train_smote,
                                                                                 sample_size=subsamples,
                                                                                 n_trees=itree_num, threshold=s0,
                                                                                 percentile=alpha)
                    X_test_new = X_test_zscore

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_zsiff, model_zsiff = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_zsiff)
                    # print("modelname_smoteiff:", modelname_smoteiff)
                    measures_zsiff = Build_Evaluation_Classification_Model(model_zsiff, X_train_new, y_train_new,
                                                                              X_test_new, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_zsiff.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_zsiff,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_zsiff, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False,
                                         ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures

def WeightedIsolationForestFilter(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    # # datasets for RQ1
    datasets = [['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv',
                 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv',
                 'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv']]

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
            print('testfile:', file_te)
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
                            print('train_file:', file_tr)
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

                    # /* step2: Zscore_SMOTE_WiFF(Zscore+smote + weighted iForest Filter) */
                    # *******************Zscore_SMOTE_WiFF*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    data_train = pd.DataFrame(np.c_[X_train_zscore, y_train])

                    # iForest_parameters = [itree_num, subsamples, hlim, mode, s0, alpha]
                    itree_num = iForest_parameters[0]
                    subsamples = iForest_parameters[1]
                    hlim = iForest_parameters[2]
                    mode_if = iForest_parameters[3]
                    s0 = iForest_parameters[4]
                    alpha = iForest_parameters[5]
                    X_train_new_idx, X_train_new, y_train_new = detect_anomalies_improved(data_train, sample_size=subsamples, n_trees=itree_num, random_state=r + 1, threshold=s0,
                                                                percentile=alpha)

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_zswiff, model_zswiff = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_zswiff)
                    # print("modelname_smotewiff:", modelname_smotewiff)
                    measures_zswiff = Build_Evaluation_Classification_Model(model_zswiff, X_train_new, y_train_new, X_test_zscore, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_zswiff.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_zswiff,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_zswiff, index=[r])
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

def WeightedIsolationForestwithLabelinformationFilter(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]
    # datasets = [['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    # # datasets for RQ1
    datasets = [['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv',
                 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv',
                 'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv']]

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
            print('testfile:', file_te)
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
                            print('train_file:', file_tr)
                            Address_tr = datapath + file_tr
                            Samples_tr = NumericStringLabel2BinaryLabel(Address_tr)  # original train data, DataFrame
                            Samples_tr.columns = column_name.tolist()  # 批量更改列名
                            Samples_tr_all = pd.concat([Samples_tr_all, Samples_tr], ignore_index=False, axis=0,
                                                       sort=False)
                    # Samples_tr_all.to_csv(f2, index=None, columns=None)  # 将类标签二元化的数据保存，保留列名，不增加行索引

                    # /* step1: data preprocessing */
                    Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug',
                                                            r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]



                    # /* step2: Zscore_SMOTE_WiFLF_CPDP(smote+weighted iForest Filter+label information) */
                    # *******************Zscore_SMOTE_WiFLF_CPDP*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    data_train = pd.DataFrame(np.c_[X_train_zscore, y_train])

                    # iForest_parameters = [itree_num, subsamples, hlim, mode, s0, alpha]
                    itree_num = iForest_parameters[0]
                    subsamples = iForest_parameters[1]
                    hlim = iForest_parameters[2]
                    mode_if = iForest_parameters[3]
                    s0 = iForest_parameters[4]
                    alpha = iForest_parameters[5]
                    X_train_new_idx, X_train_new, y_train_new = detect_anomalies_improvedplusLabel(data_train, random_state=r + 1, sample_size=subsamples, n_trees=itree_num, threshold=s0, percentile=alpha)
                    # print("X_train_new:", X_train_new_idx, X_train_new, y_train_new)
                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_wiffl, model_wiffl = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_wiffl)
                    # print("modelname_wiffl:", modelname_wiffl)
                    measures_wiffl = Build_Evaluation_Classification_Model(model_wiffl, X_train_new, y_train_new,
                                                                          X_test_zscore, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_wiffl.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_wiffl,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_wiffl, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False,
                                         ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures

def somefunc( iterable_term):

    iForest_parameters = [10, 16, 255, 'rate', 0.5, 0.5]
    runtimes = 30

    if iterable_term == 1:
        mode0 = ["null", 'M2O_CPDP', 'NM']
        for i in range(0, 3, 1):
            NaiveMethod(mode=mode0, clf_index=i, runtimes=runtimes)

    elif iterable_term == 2:
        mode1 = ['null', 'M2O_CPDP', 'ZM']
        for i in range(0, 3, 1):
            ZscoreMethod(mode=mode1, clf_index=i, runtimes=runtimes)

    elif iterable_term == 3:
        mode6 = ["null", 'M2O_CPDP', 'ZSM']
        for i in range(0, 3, 1):
            ZscoreSMOTEMethod(mode=mode6, clf_index=i, runtimes=runtimes)

    elif iterable_term == 4:
        mode9 = [iForest_parameters, 'M2O_CPDP', 'IFF']
        for i in range(0, 3, 1):
            IsolationForestFilter(mode=mode9, clf_index=i, runtimes=runtimes)

    elif iterable_term == 5:
        mode7 = [iForest_parameters, 'M2O_CPDP', 'WIFF']
        for i in range(0, 3, 1):
            WeightedIsolationForestFilter(mode=mode7, clf_index=i, runtimes=runtimes)

    elif iterable_term == 6:
        mode8 = [iForest_parameters, 'M2O_CPDP', 'WIFLF']
        for i in range(0, 3, 1):
            WeightedIsolationForestwithLabelinformationFilter(mode=mode8, clf_index=i, runtimes=runtimes)



if __name__ == '__main__':
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    starttime = time.clock()  # 计时开始

    iterable = [1, 2, 3, 4, 5, 6]  #
    pool = multiprocessing.Pool(processes=6)
    func = partial(somefunc)
    pool.map(func, iterable)
    pool.close()
    pool.join()

    endtime = time.clock()  # 计时结束

    totaltime = endtime - starttime
    print('程序结束运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序共运行时间为:', totaltime)


