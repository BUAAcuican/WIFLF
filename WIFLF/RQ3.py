#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
Answering RQ3 for Paper: How does the parameters affect on WIFLF?

@alpha: [0, 0.1, 0.2, 0.3, 0.4, 0.5]
@phi: [16, 32, 64, 128, 256, 512]
@t: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

`WIFLF: An Approach Independing The Target Project for Cross-Project Defect Prediction`
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

from weightedisolationforestpluslabel import detect_anomalies_improvedplusLabel
from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model

from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Standard_Features # Process_Data,


def WeightedIsolationForestwithLabelinformationFilter(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # datasets for RQ3
    datasets = [['arc-1.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv']]

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

def AdjustParametersofWIFLF():
    pwd = os.getcwd()
    # print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    # datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/paramResults/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    subsamplesizes = [16, 32, 64, 128, 256, 512]
    numtrees = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    alpha = []
    subsamplesize = []
    numtree = []
    mean_all = pd.DataFrame()

    for j in subsamplesizes:
        for k in numtrees:
            for i in alphas:
                iForest_parameters = [k, j, 255, 'rate', 0.5, i]  # default values of iForest
                # iForest_parameters = [100, 256, 255, 'rate', 0.5, 0.6]  # acceptable values of WIFLF
                runtimes = 30
                mode8 = [iForest_parameters, 'M2O_CPDP', 'WIFLF_params']
                df_measures = WeightedIsolationForestwithLabelinformationFilter(mode=mode8, clf_index=2, runtimes=runtimes)


                values = df_measures.iloc[:, 1:23]
                mean = pd.DataFrame(values.mean())
                mean_all = pd.concat([mean_all, mean.T])
                alpha.append(i)
                numtree.append(k)
                subsamplesize.append(j)
    mean_all['alpha'] = alpha
    mean_all['numtree'] = numtree
    mean_all['subsamplesize'] = subsamplesize
    mean_all.to_csv(spath + 'WIFLF_params.csv')

if __name__ == '__main__':
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    starttime = time.clock()  # 计时开始

    AdjustParametersofWIFLF()

    endtime = time.clock()  # 计时结束

    totaltime = endtime - starttime
    print('程序结束运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序共运行时间为:', totaltime)

