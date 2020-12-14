#!/usr/bin/env python
# -*- coding: UTF-8 -*-


"""
===================================================================================
WIFF: Weighted Isolation Forest Filter for Instance-based Cross-Project Defect
Prediction
===================================================================================

@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
===================================================================================
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

from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model

from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Process_Data, Random_Stratified_Sample_fraction, Standard_Features


def WiFF_main(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.4.csv', 'xerces-init.csv'],
    #         ['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv'],
    #         ['EQ.csv', 'JDT.csv', 'LC.csv', 'ML.csv', 'PDE.csv'],
    #         ['Apache.csv', 'Safe.csv', 'Zxing.csv']]
    datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]
    datanum = 0
    for i in range(len(datasets)):
        datanum = datanum + len(datasets[i])
    # print(datanum)
    # mode = [preprocess_mode, train_mode, save_file_name]
    preprocess_mode = mode[0]
    train_mode = mode[1]
    save_file_name = mode[2]
    iForest_parameters = mode[3]
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

                    # random sample 90% negative samples and 90% positive samples
                    string = 'bug'
                    Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string, r=r)
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接

                    target = np.c_[X_test, y_test]

                    # *******************WiFF*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    #  iForest_parameters = [itree_num, subsamples, hlim, mode, s0, alpha]
                    X_train_new, y_train_new, X_test_new, y_test_new = Process_Data(Sample_tr[:, :-1], Sample_tr[:, -1],
                                                                                    X_test, y_test, preprocess_mode,
                                                                                    iForest_parameters, r)
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_wiff, model_wiff = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_wiff)
                    # print("modelname_wiff:", modelname_wiff)
                    measures_wiff = Build_Evaluation_Classification_Model(model_wiff, X_train_new, y_train_new,
                                                                          X_test_new, y_test_new)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_wiff.update(
                        {'train_len_before': len(Sample_tr), 'train_len_after': len(Sample_tr), 'test_len': len(target),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_wiff,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_wiff, index=[r])
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


if __name__ == '__main__':

    iForest_parameters = [100, 256, 255, 'rate', 0.5, 0.1]  # default values of iForest

    mode = [[1, 0, 0, '', 'zscore', 'smote'], 'M2O_CPDP', 'eg_WiFF', iForest_parameters]

    foldername = mode[1] + '_' + mode[2]
    # only NB classifier needs to log normalization for WiFF
    WiFF_main(mode=mode, clf_index=0, runtimes=2)




