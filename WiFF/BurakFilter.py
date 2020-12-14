#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
Refer:: Turhan, B., Menzies, T., Bener, A., Di Stefano, J., 2009. On the relative value of cross-company and within-company data for defect prediction.
Empirical Software Engineering 14, 540-578. doi:10.1007/s10664-008-9103-7.
BurakFilter(BF): selecting instances from CP data for the target project
============================================================================

@Affilation：Beihang University
@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com

============================================================================
"""

# print(__doc__)

import numpy as np
import math
import pandas as pd
import math
import heapq
import random
import copy
import time
import os

from Performance import performance
from collections import Counter

# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler

from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Standard_Features, Log_transformation  # Process_Data,
from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model

def BF(X_tr, y_tr, X_te, y_te, k):
    '''
    Burak Filter/NN Filter （2009）
    使用KNN，以测试实例为驱动，为每个测试实例从训练集中选出k个最近邻点（相似点），
    即选出与测试集相似的无去重源数据实例。
    # 分类器为NB，预处理为N-->ln(N),为了避免ln0出错.将低于0.000001的值都用ln(0.000001)替换  # call self-defined fun: Log_transformation()
    :param X_tr: 二维数组，训练项目度量元数据
    :param y_tr: 一维数组，训练数据标签数据
    :param X_te: 二维数组，测试项目度量元数据
    :param y_te: 一维数组，测试数据标签数据
    :param k:  KNN中最邻近点的数量，default: k=10
    :return:  无去重的相似度训练度量元数据和标签数据
    '''
    # # only NB classifier
    # X_tr = Log_transformation(X_tr, alpha=0.000001)
    # X_te = Log_transformation(X_te, alpha=0.000001)

    # 读取源项目数据
    dataSetSize = X_tr.shape[0]
    dataSetwide = X_tr.shape[1]
    testSetSize = len(y_te)

    # 计算目标项目中每个的实例的最近邻点
    selectinstances = np.zeros(shape=(len(X_te)*k, dataSetwide))
    selectlabels = np.zeros(len(X_te)*k)  # ValueError: could not broadcast input array from shape (3) into shape (3,1)

    i = 0
    for inX in X_te:
        i += 1
        diffMat = np.tile(inX, (dataSetSize, 1)) - X_tr  # 在行上重复dataSetSize次，列上重复1次
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5  # 计算欧式距离  sqrt(Σ(x-y)^2),一个测试实例对应的距离有dataSetSize个

        # distances_te.append(distances)  # 所有的欧氏距离列表
        sortedDistIndicies = distances.argsort().tolist()  # 数组升序排序返回索引，并数组转换为列表
        selectlist = sortedDistIndicies[: k]  # 选择距离最近的k个值的索引列表
        X_tr_select = X_tr[selectlist]
        # deletlist = sortedDistIndicies[k: ]  # 删除距离远的值的索引列表
        # X_tr_select = np.delete(X_tr, deletlist, axis=0) # 与上两行结果等价
        y_tr_select = y_tr[selectlist]
        selectinstances[(i-1)*k: i*k] = X_tr_select
        selectlabels[(i-1)*k: i*k] = y_tr_select

    return selectinstances, selectlabels

def BF_main(mode, clf_index, runtimes):

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
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    # source = np.concatenate((X_train_zscore, y_train), axis=0)  # array垂直拼接
                    source = np.c_[X_train_zscore, y_train]
                    target = np.c_[X_test_zscore, y_test]


                    # /* step2: data filtering */
                    # *******************BF*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train = source[:, :-1]
                    y_train = source[:, -1]
                    # only NB classifier
                    # X_train = Log_transformation(X_train)
                    # X_test = Log_transformation(X_test)
                    X_train_new, y_train_new = BF(X_train, y_train, X_test, y_test, k=10)

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requires the label must beong to {0, 1}.
                    modelname_bf, model_bf = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_bf)
                    # print("modelname_bf:", modelname_bf)
                    measures_bf = Build_Evaluation_Classification_Model(model_bf, X_train_new, y_train_new, X_test_zscore, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time

                    measures_bf.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_bf,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_bf, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            # print('df_file_measures:\n', df_file_measures)
            # print('所有文件运行一次的结果为：\n', df_file_measures)
            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False,
                                         ignore_index=False)  # the measures of all files in runtimes
            # df_r_measures['testfile'] = file_list
            # print('df_r_measures1:\n', df_r_measures)

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures


if __name__ == '__main__':
    # X = np.array([1, 2, 3],
    #              [0, 0.00001, 2],
    #              [3, 2, 1])
    # X = np.random.randint(0, 10, (10, 2))
    # print(X)
    # x = Log_transformation(X)
    # print(x)

    mode = ['', 'M2O_CPDP', 'eg_BF']
    foldername = mode[1] + '_' + mode[2]
    # only NB classifier needs to log normalization for BF
    BF_main(mode=mode, clf_index=0, runtimes=2)