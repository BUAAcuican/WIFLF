#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
DataProcessing
============================================================================
The code is about DataProcessing.
Check_NANvalues(),
DataFrame2Array(),
NumericStringLabel2BinaryLabel(),
Random_Stratified_Sample_fraction(),
Drop_Duplicate_Samples(),
Standard_Features(),
DataImbalance(),
DevideData2TwoClasses(),
Delete_abnormal_samples(),
indexofMinMore(),
indexofMinOne(),
Process_Data(),

@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com

============================================================================
"""
# print(__doc__)

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np
import math
import heapq
import random
import copy
import time
import os
from collections import Counter
from sklearn import preprocessing, linear_model, naive_bayes, tree, neighbors, ensemble, svm, neural_network
# MinMaxScaler, StandardScaler, Binarizer, Imputer, LabelBinarizer, FunctionTransformer,model_selection, metrics,\
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# from WiFF import IForest
# from BF import BurakFilter
from scipy.spatial.distance import pdist, squareform, mahalanobis
# from __future__ import division


def Process_Data(X_train, y_train, X_test, y_test, preprocess_mode, iForest_parameters, r):

    # preprocess_mode = [drop, scale, DB, Filter, scale_mode, DB_mode]
    drop = preprocess_mode[0]
    scale = preprocess_mode[1]
    DB = preprocess_mode[2]
    Filter = preprocess_mode[3]
    scale_mode = preprocess_mode[4]
    DB_mode = preprocess_mode[5]

    # preprocess_mode = [drop, scale, DB, WiF, iF, scale_mode, DB_mode]
    # WiF = preprocess_mode[3]
    # iF = preprocess_mode[4]
    # scale_mode = preprocess_mode[5]
    # DB_mode = preprocess_mode[6]

    # iForest_parameters = [itree_num, subsamples, hlim, mode, s0, alpha]
    itree_num = iForest_parameters[0]
    subsamples = iForest_parameters[1]
    hlim = iForest_parameters[2]
    mode = iForest_parameters[3]
    s0 = iForest_parameters[4]
    alpha = iForest_parameters[5]
    data_train = np.column_stack((X_train, y_train))  # 最后一列添加数据
    check_tr_labels = list(set(y_train))
    if len(check_tr_labels) == 2:
        if drop == True:
            data_train = pd.DataFrame(data_train)
            data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
            data_train_unique = data_train_unique.values
            X_train = data_train_unique[:, : -1]
            y_train = data_train_unique[:, -1]

            # print("X1", X_train[:2, :])

        if scale == True:
            X_train, X_test = Standard_Features(X_train, scale_mode, X_test)  # data transformation
            # print("X2", X_train[:2, :])
        else:
            # print("X2'", X_train[:2, :])
            pass


        X_train_neg0, X_train_pos0, y_train_neg0, y_train_pos0 = DevideData2TwoClasses(X_train,
                                                                                       y_train)  # original X+,y+,X-,y-
        if DB == True:
            X_train_resampled, y_train_resampled = DataImbalance(X_train, y_train, r, DB_mode)  # Data imbalance
            X_train_neg, X_train_pos, y_train_neg, y_train_pos = DevideData2TwoClasses(X_train_resampled,
                                                                                       y_train_resampled)  # DB X+,y+,X-,y-
            # if WiF == True:
            if Filter == 'Weight_iForest':

                # delete noise data from negative samples by building an iForest
                # forest = IForest(100, 256, 255)  # build an iForest(t, subsample, hlim)
                # s, rate, abnormal_list = forest.Build_Forest(X_train_neg, mode='rate', r=r, s0=0.5, alpha=0.1)
                forest = IForest(itree_num, subsamples, hlim)  # build an iForest(t, subsample, hlim)
                s, rate, abnormal_list = forest.Build_Forest(X_train_neg, mode, r, s0, alpha)
                ranges_X_train_neg, Sample_X_train_neg = Delete_abnormal_samples(X_train_neg, abnormal_list)
                ranges_y_train_neg, Sample_y_train_neg = Delete_abnormal_samples(y_train_neg, abnormal_list)

                # delete noise data from positive samples by building a weight iForest
                # s_weight, rate_weight, abnormal_list_weight = forest.Build_Weight_Forest(X_train_pos0, X_train_pos,
                #                                                                          mode='rate', r=r, s0=0.50,
                #                                                                          alpha=0.1)
                s_weight, rate_weight, abnormal_list_weight = forest.Build_Weight_Forest(X_train_pos0, X_train_pos,
                                                                                         mode, r, s0, alpha)
                ranges_X_train_pos_weight, Sample_X_train_pos_weight = Delete_abnormal_samples(X_train_pos,
                                                                                               abnormal_list_weight)
                ranges_y_train_pos_weight, Sample_y_train_pos_weight = Delete_abnormal_samples(y_train_pos,
                                                                                               abnormal_list_weight)
                X_train = np.concatenate((Sample_X_train_neg, Sample_X_train_pos_weight), axis=0)  # array垂直拼接
                y_train = np.array(Sample_y_train_neg.tolist() + Sample_y_train_pos_weight.tolist())  # list垂直拼接

            elif Filter == 'iForest':
            # elif (WiF == False) and (iF == True):
                # delete noise data from negative samples by building an iForest
                # forest = IForest(100, 256, 255)  # build an iForest(t, subsample, hlim)
                # s, rate, abnormal_list = forest.Build_Forest(X_train_neg, mode='rate', r=5, s0=0.5, alpha=0.1)
                forest = IForest(itree_num, subsamples, hlim)  # build an iForest(t, subsample, hlim)
                s, rate, abnormal_list = forest.Build_Forest(X_train_neg, mode, r, s0, alpha)
                ranges_X_train_neg, Sample_X_train_neg = Delete_abnormal_samples(X_train_neg, abnormal_list)
                ranges_y_train_neg, Sample_y_train_neg = Delete_abnormal_samples(y_train_neg, abnormal_list)

                # delete noise data from positive samples by building an iForest
                # s_pos, rate_pos, abnormal_list_pos = forest.Build_Forest(X_train_pos, mode='rate', r=5, s0=0.50,
                #                                                          alpha=0.1)
                s_pos, rate_pos, abnormal_list_pos = forest.Build_Forest(X_train_pos, mode, r, s0, alpha)
                ranges_X_train_pos, Sample_X_train_pos = Delete_abnormal_samples(X_train_pos,
                                                                                        abnormal_list_pos)
                ranges_y_train_pos, Sample_y_train_pos = Delete_abnormal_samples(y_train_pos,
                                                                                        abnormal_list_pos)
                X_train = np.concatenate((Sample_X_train_neg, Sample_X_train_pos), axis=0)  # array垂直拼接
                y_train = np.array(Sample_y_train_neg.tolist() + Sample_y_train_pos.tolist())  # list垂直拼接

            elif Filter == 'Burak_Filter':
                X_train,  y_train = BF(X_train, y_train, X_test, y_test, k=10)

            elif Filter == 'HISNN':
                X_train, y_train = HISNN(X_train, y_train, X_test, y_test, k=10)

            elif Filter == 'HSBF':

                X_train, y_train = HSBF(X_train, y_train, X_test, y_test, k=10)

            elif Filter == 'DFAC':
                X_train, y_train = DFAC(X_train, y_train, X_test, y_test)

            elif Filter == 'None':
                X_train = X_train_resampled
                y_train = y_train_resampled

        else:
            if Filter == 'iForest':  # iF == True:
                # delete noise data from negative samples by building an iForest
                # forest = IForest(100, 256, 255)  # build an iForest(t, subsample, hlim)
                # s, rate, abnormal_list = forest.Build_Forest(X_train_neg0, mode='rate', r=5, s0=0.5, alpha=0.1)
                forest = IForest(itree_num, subsamples, hlim)  # build an iForest(t, subsample, hlim)
                s, rate, abnormal_list = forest.Build_Forest(X_train_neg0, mode, r, s0, alpha)
                ranges_X_train_neg, Sample_X_train_neg = Delete_abnormal_samples(X_train_neg0, abnormal_list)
                ranges_y_train_neg, Sample_y_train_neg = Delete_abnormal_samples(y_train_neg0, abnormal_list)

                # delete noise data from positive samples by building an iForest
                # s_pos, rate_pos, abnormal_list_pos = forest.Build_Forest(X_train_pos0, mode='rate', r=5, s0=0.50,
                #                                                          alpha=0.1)
                s_pos, rate_pos, abnormal_list_pos = forest.Build_Forest(X_train_pos0, mode, r, s0, alpha)
                ranges_X_train_pos, Sample_X_train_pos = Delete_abnormal_samples(X_train_pos0,
                                                                                 abnormal_list_pos)
                ranges_y_train_pos, Sample_y_train_pos = Delete_abnormal_samples(y_train_pos0,
                                                                                 abnormal_list_pos)
                X_train = np.concatenate((Sample_X_train_neg, Sample_X_train_pos), axis=0)  # array垂直拼接
                y_train = np.array(Sample_y_train_neg.tolist() + Sample_y_train_pos.tolist())  # list垂直拼接

            elif Filter == 'Burak_Filter':
                X_train, y_train = BF(X_train, y_train, X_test, y_test, k=10)

            elif Filter == 'HISNN':
                X_train, y_train = HISNN(X_train, y_train, X_test, y_test, k=10)

            elif Filter == 'HSBF':

                X_train, y_train = HSBF(X_train, y_train, X_test, y_test, k=10)

            elif Filter == 'DFAC':
                X_train, y_train = DFAC(X_train, y_train, X_test, y_test)

            elif Filter == 'iForest_Filter':  # apply original iForest
                X_train0 = np.concatenate((X_train_neg0, X_train_pos0), axis=0)  # array垂直拼接
                y_train0 = np.array(y_train_neg0.tolist() + y_train_pos0.tolist())  # list垂直拼接
                forest = IForest(itree_num, subsamples, hlim)  # build an iForest(100, 256, 255)
                s, rate, abnormal_list = forest.Build_Forest(X_train0, mode, r, s0, alpha)
                ranges_X_train, X_train = Delete_abnormal_samples(X_train0, abnormal_list)
                ranges_y_train, y_train = Delete_abnormal_samples(y_train0, abnormal_list)
            elif Filter == 'None':
                X_train = np.concatenate((X_train_neg0, X_train_pos0), axis=0)  # array垂直拼接
                y_train = np.array(y_train_neg0.tolist() + y_train_pos0.tolist())  # list垂直拼接


            else:
                pass

        return X_train, y_train, X_test, y_test

    else:
        return None




def DataImbalance(X, y, r, mode='smote'):
    '''
    To sample X×y, and make the majority and the minority balanced
    :param X: 2D array
    :param y: 1D array
    :param mode: default: 'smote', the method to solve data imbalance issue. the parameters are
                 'smote', 'rus', 'ros',  type: string.
    :return: resampled X (2D array) and resampled y (1D array)
    '''

    y_label_number = Counter(y)
    # print('data distribution:', y_label_number)
    # print(type(y), y_label_number)  # Counter({1: 163, 0: 305})

    if mode == 'ros':
        print('......ROS:Data imbalance processing......')
        # 使用RandomOverSampler从少数类的样本中进行随机采样来增加新的样本使各个分类均衡
        ros = RandomOverSampler(random_state=r + 1)
        X_resampled_ros, y_resampled_ros = ros.fit_sample(X, y)
        y_ros = sorted(Counter(y_resampled_ros).items())
        # print(y_ros)
        return X_resampled_ros, y_resampled_ros

    elif mode == 'smote':
        print('......SMOTE:Data imbalance processing......')
        # SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本
        X_resampled_smote, y_resampled_smote = SMOTE(random_state=r + 1).fit_sample(X, y)
        y_smote = sorted(Counter(y_resampled_smote).items())
        # print(y_smote)
        return X_resampled_smote, y_resampled_smote

    elif mode == 'rus':
        print('......RUS:Data imbalance processing......')
        # RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据: 随机选取数据的子集.
        rus = RandomUnderSampler(random_state=r + 1)
        X_resampled_rus, y_resampled_rus = rus.fit_sample(X, y)
        y_rus = sorted(Counter(y_resampled_rus).items())
        # print(y_rus)
        return X_resampled_rus, y_resampled_rus

    else:
        print('Check the value of the parameter:mode')
        return ValueError

def DataFrame2Array(file):
    """
    Transfer DataFrame to Array.
    :param file:  absolute file name, type: string.
    :param X_te: A 2-D arraythat denotes test instances.
    :return: X——2-D array, type is float32;
             y——1-D array, type is float32;
    """
    df = pd.read_csv(file, engine='python')  # 读取训练集(缺陷类)
    col_name = df.columns.values
    # label = df[col_name[-1]].unique()
    # print(label, label.shape[0])
    X = df[col_name[:-1]].values
    X.astype(np.float32)
    # print(X.shape, type(X))  # 查看数据类型，及样本量和特征数量
    y = df[col_name[-1]].values
    # y = y.reshape(-1, 1)  # 将一维向量转换为数组
    return X, y

def DevideData2TwoClasses(X, y):
    '''
    devide data including 2 classes to 2 classed, return each class data.
    :param X: 2d Array, including 2 class samples.
    :param y: 1d array or a list, class label values
    :return: X_neg, y_neg: negative samples (features:X_neg, class labels: y_neg)
             X_pos, y_pos: positive samples (features:X_pos, class labels: y_pos)
    '''

    # print('......Devide data to 2 parts......')
    X = list(X)
    X_neg = []
    y_neg = []
    X_pos = []
    y_pos = []
    for i in range(len(y)):
        if y[i] == 0:
            X_neg.append(X[i])   # 正样本的数量
            y_neg.append(y[i])
        elif y[i] > 0:
            X_pos.append(X[i])  # 负样本的数量
            y_pos.append(y[i])
        else:
            print('the data type is wrong, please check...')
    X_neg = np.array(X_neg)
    y_neg = np.array(y_neg)
    X_pos = np.array(X_pos)  # list2Array
    y_pos = np.array(y_pos)

    if len(X_neg) + len(X_pos) == len(y) and len(y_neg) + len(y_pos) == len(y):
        return X_neg, X_pos, y_neg, y_pos
    else:
        raise ValueError('Devide error!')
        # return None

def Delete_abnormal_samples(Sample, abnormal_list):
    '''
    delete abnormal samples from the original sample, return the normal samples.
    :param Sample: 2D array
    :param abnormal_list: the index about abnormal samples in Sample
    :return: ranges, type: list, the left sample index in original Sample
             Sample, type: 2D array, the left samples after deleting abnormal samples
    '''
    Sample = Sample.tolist()
    ranges = list(range(len(Sample)))

    delete_sample = []
    # while abnormal_list:
    if abnormal_list != []:
        for i in abnormal_list:
            ranges.remove(i)
            delete_sample.append(Sample[i])

    # while delete_sample:
    if delete_sample != []:
        for j in delete_sample:
            Sample.remove(j)
    Sample = np.array(Sample)

    return ranges, Sample

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
    print('----Drop duplicate samples...')
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

def Check_NANvalues(fpath):
    '''

    :param fpath:
    :return:
    '''
    fl = os.listdir(fpath)
    nanf_num = 0  # 初始化存在空值的文件数量

    for f in fl:
        # print('====检测数据集是否有缺失值=====')
        df = pd.read_csv(fpath + '/' + f)
        column = df.columns.values  # 列名
        colisnull = df.isnull().values.any()  # 判断各个列是否有缺失值
        if colisnull == True:
            nanf_num += 1
            coltotalnull = df.isnull().sum()  # 各列的缺失值总数
            for columname in column:
                isnan = df[columname].isnull().any()  # 判断某列是否有缺失值
                if isnan:
                    allnan = df[columname].isnull().all()  # 判断某列是否全部为缺失值
                    if allnan:
                        print('列名："{}",全部为缺失值'.format(columname), '\n')
                    else:
                        nalist = [df[columname].isnull().values == True]  # 判断为某列的值是否为空值，返回bool列表
                        # print(type(nalist))  # 返回列表
                        for na in nalist:
                            ind = np.where(na)[0]  # 数组值的位置索引np.where(na)，返回元组,元组的原始索引np.where(na)[0]
                            print('列名："{}", 第{}行位置有缺失值'.format(columname, ind), '\n')
                            # print(type(ind))  # 返回数组

                else:
                    print('列名："{}",没有缺失值'.format(columname), '\n')
            print('文件中各列的缺失值情况：\n', coltotalnull, '\n')
        else:
            # print('%s文件中没有缺失值' % f.rstrip('.csv'), '\n')
            pass
    print('%d个文件存在缺失值\n' % (nanf_num))
    return nanf_num

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
        print('----Do Z-score on source and target datasets according to source ...')
        zscore_scaler = preprocessing.StandardScaler(copy=True, with_mean=True,
                                                     with_std=True)  # Binarizer, Imputer, LabelBinarizer
        zscore_scaler.fit(X_train)
        X_train_zscore = zscore_scaler.transform(X_train)
        X_test_zscore = zscore_scaler.transform(X_test)
        return X_train_zscore, X_test_zscore

    # 规范化方法：max-min normalization，原始数据的线性变换，使结果映射到[0,1]区间
    # 实现特征极小方差的鲁棒性以及在稀疏矩阵中保留零元素。（鲁棒性：表征控制系统对特性或参数扰动的不敏感性）
    elif mode == 'maxmin':
        print('----Do max-min on source and target datasets  according to source ...')
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(X_train)
        X_test_minmax = min_max_scaler.transform(X_test)
        return X_train_minmax, X_test_minmax
    if mode == 'zscore_t':
        print('----Do Z-score on source and target datasets according to target ...')
        zscore_scaler = preprocessing.StandardScaler(copy=True, with_mean=True,
                                                     with_std=True)  # Binarizer, Imputer, LabelBinarizer
        zscore_scaler.fit(X_test)
        X_test_zscore = zscore_scaler.transform(X_test)
        X_train_zscore = zscore_scaler.transform(X_train)
        return X_train_zscore, X_test_zscore

        # 规范化方法：max-min normalization，原始数据的线性变换，使结果映射到[0,1]区间
        # 实现特征极小方差的鲁棒性以及在稀疏矩阵中保留零元素。（鲁棒性：表征控制系统对特性或参数扰动的不敏感性）
    elif mode == 'maxmin_t':
        print('----Do max-min on source and target datasets according to target ...')
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
        print('----Do score(mean=0, std=1) on source and target separately....')
        X_train_scaled = preprocessing.scale(X_train)
        X_test_scaled = preprocessing.scale(X_test)
        return X_train_scaled, X_test_scaled

    elif mode == 'maxminscale':
        print('----Do max-min on source and target separately....')
        X_train_minmaxscaled = preprocessing.MinMaxScaler().fit_transform(X_train)
        X_test_minmaxscaled = preprocessing.MinMaxScaler().fit_transform(X_test)
        return X_train_minmaxscaled, X_test_minmaxscaled

    elif mode == 'logscale':
        print('----Do log(x+1) on source and target separately....')
        X_train_logscaled = preprocessing.FunctionTransformer(np.log1p, validate=True).fit_transform(X_train)  # log1p = log(1+x)
        X_test_logscaled = preprocessing.FunctionTransformer(np.log1p, validate=True).fit_transform(X_test)
        return X_train_logscaled, X_test_logscaled

    else:
        raise ValueError('the value of mode is wrong, please check...')
        # return None

def Log_transformation(X, alpha=0.000001):
    '''
    将N-->ln(N),为了避免ln0出错.将低于0.000001的值都用ln(0.000001)替换
    :param X: 2-D array, mxn,m:rows, n:columns
    :param alpha: float, a numeric value, default value: alpha = 0.000001
    :return:  X_log: 2-D array, mxn,m:rows, n:columns
    '''
    X_log = np.zeros((X.shape[0], X.shape[1]))
    for i in range(len(X)):
        for j in range(X.shape[1]):
            temp = X[i][j]
            if temp < alpha:
                X_log[i][j] = np.log(alpha)
            else:
                X_log[i][j] = np.log(temp)
    return X_log

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

def indexofMinOne(arr):
    '''

    :param arr: type: list
    :return: the index of min value, only return the first min value
    '''
    minindex = 0
    currentindex = 1
    while currentindex < len(arr):
        if arr[currentindex] < arr[minindex]:
            minindex = currentindex
        currentindex += 1
    return minindex

def indexofMinMore(arr):
    '''

    :param arr: type: list
    :return: the index of min value, only return  all the min value
    '''

    minvalue = min(arr)
    minindex = []
    currentindex = 0
    while currentindex < len(arr):
        if arr[currentindex] == minvalue:
            minindex.append(currentindex)
        currentindex += 1
    return minindex

def mashi_distance(x, y):
    # 马氏距离要求样本数要大于维数，否则无法求协方差矩阵
    # 此处进行转置，表示10个样本，每个样本2维
    X = np.vstack([x, y])
    # print('X:', X, X.shape)
    XT = X.T
    # print('XT:', XT.shape)

    # 方法一：根据公式求解
    S = np.cov(X)  # 两个维度之间协方差矩阵
    # print('S:', S)
    SI = np.linalg.inv(S)  # 协方差矩阵的逆矩阵
    # 马氏距离计算两个样本之间的距离，此处共有10个样本，两两组合，共有45个距离。
    n = XT.shape[0]
    d1 = []
    for i in range(0, n):
        for j in range(i + 1, n):
            delta = XT[i] - XT[j]
            d = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
            d1.append(d)
    return d1

def get_keys(d, value):
    return [k for k, v in d.items() if v == value]

def get_values(d, key):
    return [v for k, v in d.items() if k == key]


if __name__ == '__main__':

    # x = np.random.random(10)
    # y = np.random.random(10)
    # d1 = mashi_distance(x, y)
    # print(d1, len(d1))
    # x = mvnrnd([0;0], [1 .9;.91], 100)#样本数据
    # y = np.array([1,1],[1,- 1][-1, 1],[-1, - 1])  #  观测数据
    # fpath = r'D:\PycharmProjects\cuican\data2020\AEEEM\data'
    # filelist = os.listdir(fpath)

    fpath = r'D:\PycharmProjects\cuican\data2020\AEEEM\data'
    # filelist = os.listdir(fpath)
    # for file in filelist:
    #     df_X = NumericStringLabel2BinaryLabel(file)

    df = pd.read_csv(fpath + '\\ML.csv')
    Sample = df.values
    X_train = Sample[:, :-1]
    y_train = Sample[:, -1]
    X_test = Sample[:10, :-1]
    y_test = Sample[:10, -1]
    preprocess_mode = [1, 0, 0, 'none', 'zscore', 'smote']
    iForest_parameters = [100, 256, 255, 'rate', 0.5, 0.1]
    Process_Data(X_train, y_train, X_test, y_test, preprocess_mode, iForest_parameters, r = 2)




    # lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))






