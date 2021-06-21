#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
TCA+: Reference: Transfer Defect Learning-P41-2013
============================================================================
@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
@2019/12/30
============================================================================
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
# from sklearn.merics.pairwise import euclidean_distances,
from scipy.spatial.distance import pdist, mahalanobis
from scipy.linalg import det
from collections import Counter
from sklearn import preprocessing, model_selection, metrics,\
    linear_model, naive_bayes, tree, neighbors, ensemble, svm, neural_network
# MinMaxScaler, StandardScaler, Binarizer, Imputer, LabelBinarizer, FunctionTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# write by Cuican
from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model


from TCA import TCA

def TCAplus(source,target):
    '''
    TCA+ Summary of this function goes here
     Reference: Transfer Defect Learning-P41-2013
     Detailed explanation goes here:
    :param source: each row is a observation, the last column is the label {0,1}
    :param target: each row is a observation, the last column is the label {0,1}
    :return: measures
    '''
    # global t_train
    # global t_test, tic


    # The number of instances and features
    numRow1 = source.shape[0]
    numcol1 = source.shape[1]
    numRow2 = target.shape[0]
    numcol2 = target.shape[1]

    source_x = source[:, :-1]
    target_x = target[:, :-1]

    DIST_source = []
    DIST_target = []
    ## Sep1:Calculate Euclidean distance (DIST) for source data and target data, respectively.
    for i in range(numRow1-1):  # each of the first (numRow1-1) instances
        for j in range(i, numRow1):  # each of the last (numRow1-i) instances
            DIST_source.append(np.linalg.norm(source_x[i,:]-source_x[j,:], 2))

    for i in range(numRow2-1):
        for j in range(i, numRow2):
            DIST_target.append(np.linalg.norm(target_x[i,:]-target_x[j,:], 2))

    ## Step2: Calaulate DCV: 6 elements of a characteristic vector for a data set
    # DCV_source = np.zeros(6, 1)  # [min, max, mean, medium, std, number of Rows]
    # DCV_target = np.zeros(6, 1)

    DCV_source = [np.min(DIST_source), np.max(DIST_source), np.mean(DIST_source),
                  np.median(DIST_source), np.std(DIST_source), numRow1]
    DCV_target = [np.min(DIST_target), np.max(DIST_target), np.mean(DIST_target),
                  np.median(DIST_target), np.std(DIST_target), numRow2]

    # # Step3: Similarity vector and Normalization selections
    X_train_new = np.zeros((numRow1, numcol1 - 1))  # initialization, global variable
    X_test_new = np.zeros((numRow2, numcol2 - 1))  # initialization, global variable
    # rule = 1: dist_mean && dist_std:same -->  no normalization
    if (DCV_source[2] * 0.9 <= DCV_target[2] <= DCV_source[2] * 1.1) and \
            (DCV_source[4] * 0.9 < DCV_target[4] <= DCV_source[4] * 1.1):
        X_train_new = source_x
        X_test_new = target_x
    # rule = 2: dist_max && dist_min && numRow: much less or much more # Perform max-min normalization
    elif((DCV_target[0] < DCV_source[0] * 0.4) and (DCV_target[1] < DCV_source[1] * 0.4) and (DCV_target[5] < DCV_source[5] * 0.4)) \
            or ((DCV_source[0] * 1.6 < DCV_target[0]) and (DCV_source[1] * 1.6 < DCV_target[1]) and (DCV_source[5] * 1.6 < DCV_target[5])):
        X_train_new, X_test_new = Standard_Features(source_x, 'maxminscale', target_x)
    # rule = 3   dist_std:much more &&numRow2 < numRow1; dist_std:much less &&numRow2 > numRow1 # Only perform normalization on source data
    elif ((DCV_source[4] * 1.6 < DCV_target[4]) and (numRow2 < numRow1)) or (
                (DCV_target[4] < DCV_source[4] * 0.4) and (numRow2 > numRow1)):
        # Error:Attempted to access DCV_source(5) index out of bounds because numel(DCV_source)=4.
        X_train_new = preprocessing.scale(source_x)
        X_test_new = target_x
    # rule = 4  dist_std:much more &&numRow2 > numRow1; dist_std:much less &&numRow2 < numRow1# Only perform normalization on target data
    elif ((DCV_source[4] * 1.6 < DCV_target[4]) and (numRow2 > numRow1)) or (
                (DCV_target[4] < DCV_source[4] * 0.4) and (numRow2 < numRow1)):
        X_test_new = preprocessing.scale(target_x)
        X_train_new = source_x
    # rule = 5  # Z-score normalization
    else:
        X_train_new, X_test_new = Standard_Features(source_x, 'scale', target_x)
    # print(X_train_new.shape, X_test_new.shape)

    # Calculate the components
    # source_x_new, target_x_new = TCA(X_train_new, X_test_new)  # Call TCA()

    tca = TCA(kernel_type='linear', dim=15, lamb=1, gamma=1)
    source_x_new, target_x_new = tca.fit(X_train_new, X_test_new)
    # print('shape:', source_x_new.shape, target_x.shape)
    # Component based source data and target data
    tca_source = np.c_[source_x_new, source[:, -1]]
    tca_target = np.c_[target_x_new, target[:, -1]]
    print('after TCA+:', tca_source.shape, tca_target.shape)
    return tca_source, tca_target



def NumericStringLabel2BinaryLabel(file):
    '''
    Transfer the numeric or string label to binary label. return the transfered values, type: DataFrame
    :param file: file name (absolute path), type:string;
    :return: df_X, type: DataFrame, The values after transfering.
    '''

    df = pd.read_csv(file, engine='python')
    column = df.columns.values  # column name
    # print('the number of metrics is:', len(column) - 1, '\n',
    #       'the metrics are :', column)

    # print('......Label binarization......')
    df_y = df[column[-1]]  # read the last column(i.e., class labels)
    df_X = df[column[:-1]]  # read metrics
    # print('Check the value of the label:', df_y.iloc[0])

    # for i in range(len(df_y)):
    #     if df_y.iloc[i] == 'buggy' or df_y.iloc[i] == 'yes' or df_y.iloc[i] == 'Y':
    #         df.loc[i, 'bug'] = 1
    #         # df_y.iloc[i] = 1
    #     elif df_y.iloc[i] == 'clean' or df_y.iloc[i] == 'no' or df_y.iloc[i] == 'N':
    #         # df_y.iloc[i] = 0
    #         df.loc[i, 'bug'] = 0
    #
    #     else:
    #         if df_y.iloc[0] >= 0:
    #             df.loc[i, 'bug'] = 1
    #         else:
    #             df.loc[i, 'bug'] = 0
    # df.drop(columns=column.tolist()[-1], inplace=True)
    # return df

    if not isinstance(df_y.iloc[0], str):
        if (df_y.iloc[0] >= 0):
            cb_numeric = (df_y.values > 0).astype(np.int32)
            df_X['bug'] = cb_numeric  # add a column named 'bug'
    else:

        classbinarizer = preprocessing.LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
        cb_string = classbinarizer.fit_transform(df_y.values)  # 少类为0，多类为1，与预测相反，因此需要调整
        df_X['bug'] = (cb_string == 0).astype(np.int32)  # add a column named 'bug'

    # print('Check the results of label transformation: \n', df_X.head(3))  #
    if df_X.shape == df.shape:
        # print('postive_num:', df_X['bug'].sum())
        return df_X
    else:
        # print(' The shapes of %s before and after transformation are different ' % (file))
        return AttributeError


def Random_Stratified_Sample_fraction(data, string, r):
    '''
    Random Stratified Sample according to some fractions according to rows
    :param data: type: DataFrame, mxn, m rows, n columns
    :param string: type: str. stratified according a cloumn named string.
    :param r: random state, 1~255
    :return: X_pos: type: DataFrame, the selected positive sample
             X_neg: type: DataFrame, the selected negative sample
             X_pos_index: type: list, the selected positive sample index
             X_neg_index: type: list, the selected negative sample index
    '''
    gbr = data.groupby(string, group_keys=0)  # Grouping criteria are not shown
    X_pos = pd.DataFrame()
    X_neg = pd.DataFrame()
    X_pos_index = []
    X_neg_index = []
    for name, group in gbr:
        # print('name:', name)  # one of the grouping name in group
        # print('group:', group)  # values of the group, original values according to string
        typicalFracDict = {1: 0.9, 0: 0.9}
        frac = typicalFracDict[name]
        # random sample according to the row
        result = group.sample(n=None, frac=frac, replace=False, weights=None, random_state=r+1, axis=0)
        if name == 1:
            # frac_bug = len(result) / len(group)   # bug_num/ bug_num_ori≈frac
            X_pos = result
            X_pos_index = X_pos.index.values
            X_pos_index.sort()  # ranking from small to large
        else:
            # frac_nbug = len(result) / len(group)  # nbug_num/ nbug_num_ori≈frac
            X_neg = result
            X_neg_index = X_neg.index.values
            X_neg_index.sort()  # ranking from small to large
        # print("name:", name, '\n result:\n', result)

    return X_pos, X_neg, X_pos_index, X_neg_index

def Drop_Duplicate_Samples(data):
    '''
    drop duplicate samples of data
    :param data: type: DataFrame, m X n, m rows, n columns
    :return: data_unique: type: DataFrame, m' X n, m' rows, n columns, m'<= m
    '''
    print('*                  Drop duplicate samples...')
    data_unique = data.drop_duplicates(subset=None, keep='first', inplace=False)  # 默认值，删除重复行，只保留第一次，在副本上操作
    # data_unique.to_csv(spath + '\\' + f, index=None, columns=None)  # 没有重复数据，原数据保存，保留列名，不增加行索引
    # data_unique = data_unique.reset_index(drop=True)  # 重新设置行索引，并将旧的行索引删除
    # data_unique.to_csv(spath1 + '\\' + f, index=None, columns=None)  # 没有重复数据，原数据保存，保留列名，不增加行索引
    # if len(data) == len(data_unique):
    #     print('No duplicate samples:%d\n' % (len(data_unique)))
    #     pass
    # else:
    #     print('Before dropping duplicate samples:', len(data),
    #           ';After dropping duplicate samples:', len(data_unique), '\n')  # '文件的有重复样本值已删除\n'
    return data_unique


def Standard_Features(X_train, mode , X_test):
    '''
    :param X_train: type: dataFrame or 2D-array, mXn, m rows, n columns: n features
    :param X_test: type: dataFrame or 2D-array, m'Xn', m' rows, n' columns: n' features.
                   when only X_train is scaled (i.e., mode='scale'), X_test does not must be input.
    :param mode: process feature values modes: 'zscore', 'maxmin', 'log', 'scale'
    :return: Standard_X_train: 2d Array, m X n, m rows, n columns: n features
             Standard_X_test: 2d Array, m' X n', m' rows, n' columns: n' features
    '''

    # 每个项目都使用自身的均值方差/不使用归一化，是考虑测试集可能不包括最值，归一化可以用来和标准化比较影响
    # CC认为即使改变数据分布也没关系，因为跨项目数据分布本来就和源数据分布不一致
    # Xy = Sample.values  # DataFrame2Array
    # X = Sample[:, :-1]  # mX(n-1): m samples, n features
    # y = Sample[:, -1]  # mX1: m label values
    # X_Mean = X.mean(axis=0)
    # X_Std = X.std(axis=0)
    # X_Max = X.max(axis=0)
    # X_Min = X.min(axis=0)

    # 正规化方法:z-score, 均值为0，方差为1， 适用于属性A的最大值和最小值未知的情况，或有超出取值范围的离群数据的情况。
    if mode == 'zscore':
        print('*                  Do Z-score on source & target datasets according to source ...')
        zscore_scaler = preprocessing.StandardScaler(copy=True, with_mean=True,
                                                     with_std=True)  # Binarizer, Imputer, LabelBinarizer
        zscore_scaler.fit(X_train)
        X_train_zscore = zscore_scaler.transform(X_train)
        X_test_zscore = zscore_scaler.transform(X_test)
        return X_train_zscore, X_test_zscore

    # 规范化方法：max-min normalization，原始数据的线性变换，使结果映射到[0,1]区间
    # 实现特征极小方差的鲁棒性以及在稀疏矩阵中保留零元素。（鲁棒性：表征控制系统对特性或参数扰动的不敏感性）
    elif mode == 'maxmin':
        print('*                  Do max-min on source & target datasets according to source ...')
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(X_train)
        X_test_minmax = min_max_scaler.transform(X_test)
        return X_train_minmax, X_test_minmax
    if mode == 'zscore_t':
        print('*                  Do Z-score on source & target datasets according to target ...')
        zscore_scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)  # Binarizer, Imputer, LabelBinarizer
        zscore_scaler.fit(X_test)
        X_test_zscore = zscore_scaler.transform(X_test)
        X_train_zscore = zscore_scaler.transform(X_train)
        return X_train_zscore, X_test_zscore

        # 规范化方法：max-min normalization，原始数据的线性变换，使结果映射到[0,1]区间
        # 实现特征极小方差的鲁棒性以及在稀疏矩阵中保留零元素。（鲁棒性：表征控制系统对特性或参数扰动的不敏感性）
    elif mode == 'maxmin_t':
        print('*                  Do max-min on source and target datasets according to target ...')
        min_max_scaler = preprocessing.MinMaxScaler()
        X_test_minmax = min_max_scaler.fit_transform(X_test)
        X_train_minmax = min_max_scaler.transform(X_train)
        return X_train_minmax, X_test_minmax

    elif mode == 'log':
        if X_train.any() >= 0:  #  ?? 为何有负数还为True
            #
            # print('Do log(x+1) on source and target datasets...')
            log_scaler = preprocessing.FunctionTransformer(np.log1p, validate=True)  # log1p = log(1+x)
            X_train_log = log_scaler.fit_transform(X_train)
            if X_test.all() >= 0:
                X_test_log = log_scaler.transform(X_test)
                return X_train_log, X_test_log
            else:
                raise ValueError('test data exists negative values')
                # return None

        else:
            raise ValueError('training data exists negative values')
            # return None

    elif mode == 'scale':
        print('*                  Do score(mean=0, std=1) on source and target separately....')
        X_train_scaled = preprocessing.scale(X_train)
        X_test_scaled = preprocessing.scale(X_test)
        return X_train_scaled, X_test_scaled

    elif mode == 'maxminscale':
        print('*                  Do max-min on source and target separately....')
        X_train_minmaxscaled = preprocessing.MinMaxScaler().fit_transform(X_train)
        X_test_minmaxscaled = preprocessing.MinMaxScaler().fit_transform(X_test)
        return X_train_minmaxscaled, X_test_minmaxscaled

    elif mode == 'logscale':
        print('*                  Do log(x+1) on source and target separately....')
        X_train_logscaled = preprocessing.FunctionTransformer(np.log1p, validate=True).fit_transform(X_train)  # log1p = log(1+x)
        X_test_logscaled = preprocessing.FunctionTransformer(np.log1p, validate=True).fit_transform(X_test)
        return X_train_logscaled, X_test_logscaled

    else:
        raise ValueError('the value of mode is wrong, please check...')
        # return None


def TCAplus_main(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # datasets for RQ2
    datasets = [
        ['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv',
         'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv',
         'xalan-2.4.csv', 'xerces-init.csv'],
        ['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv', 'lucene-2.0.csv',
         'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv',
         'xalan-2.6.csv', 'xerces-1.3.csv'],
        ['EQ.csv', 'JDT.csv', 'LC.csv', 'ML.csv', 'PDE.csv'],
        ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    # datasets for example
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

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

                    # Train，build and evaluate model: model requires the label must beong to {0, 1}.
                    source_new, target_new = TCAplus(source, target)
                    # Train model: classifier / model requires the label must beong to {0, 1}.
                    modelname, model = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname)
                    # print("modelname:", modelname)
                    measures = Build_Evaluation_Classification_Model(model, source_new, source_new[:, -1],
                                                                     target_new,
                                                                     target_new[:, -1])  # Training and Prediction
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures.update({'train_len_before': len(Sample_tr),
                                     'train_len_after': len(source),
                                     'test_len': len(target),
                                     'runtime': run_time,
                                     'clf': clf_index,
                                     'testfile': file_te,
                                     'trainfile': 'More1'})
                    df_m2ocp_measures = pd.DataFrame(measures, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_file_measures = pd.concat([df_file_measures, df_m2ocp_measures], axis=0, sort=False,
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

    mode = ['null', 'M2O_CPDP', 'eg_TCA+']
    for i in range(0, 1, 1):
        TCAplus_main(mode=mode, clf_index=i, runtimes=30)