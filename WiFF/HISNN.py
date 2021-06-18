#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
HISNN: Hybrid Instance Selection Using Nearest-Neighbor
Reference: Duksan Ryu et al.: A Hybrid Instance Selection Using NN for Cross-Project Defect Prediction.
Journal of Computer Science and Technology
============================================================================
@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
@2019/12/20
============================================================================
"""
# print(__doc__)

# from sklearn.merics.pairwise import euclidean_distances,
from scipy.spatial.distance import pdist, mahalanobis
from scipy.linalg import det
from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model
from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Standard_Features, Log_transformation  # Process_Data,
from Performance import performance
import math
import heapq
import random
import copy
import numpy as np
import time
import os
import pandas as pd
from collections import Counter
from sklearn import preprocessing, model_selection, metrics,\
    linear_model, naive_bayes, tree, neighbors, ensemble, svm, neural_network
# MinMaxScaler, StandardScaler, Binarizer, Imputer, LabelBinarizer, FunctionTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def mahal(Y, X):
    '''
     度量一个样本点y与数据分布为X的集合的距离
    :param Y: 观测数据 Y=[y1,y2,...,ym], Y为m×n的矩阵，m为样本数，n为特征数，y为一个向量1×n
    :param X: 样本数据 X = [x1,x2,...,xk]，X为k×n的矩阵，k为样本数，n为特征数，x为一个向量1×n
    :return: dist: list, Y中每个样本到集合X的马氏距离。
    '''
    ry = Y.shape[0]
    cy = Y.shape[1]
    u = np.mean(X, axis=0)  # mean value of samples
    # print('u:', u)
    XT = X.T
    sigma = np.cov(XT)  # covariance matrix: 特征和特征之间的
    # print('sigma:', sigma, sigma.shape)
    try:
       sigma_inv = np.linalg.inv(sigma)  # the inverse of covariance matrix
       # print('s_i:', np.linalg.inv(sigma))
    except:
        print("Inverse matrix does not exist!")
        raise ValueError('Singular matrix')
    dist = []
    for i in range(len(Y)):
        y = Y[i, :]   # y-u：[0.3 3.1 2.9] (3,)
        delta = (y - u).reshape(1, -1)  # delta: [[0.3 3.1 2.9]] (1,3)
        # print('delta:', delta, delta.shape)
        di = np.dot(np.dot(delta, sigma_inv), (delta.T) )
        # print('di:', di)
        di = di.tolist()
        dist.append(di[0][0])

    # delta = Y - np.tile(u, (ry, 1))
    # print('delta', delta)
    # m = delta * sigma_inv
    # # m = np.dot(delta.T, sigma_inv)
    # print('m:', m)
    # sqr = m * ( delta.T)
    # # sqr = np.dot(m, delta)
    # print('sqr:', sqr)
    # dist = np.sqrt(sqr)
    # print(dist)

    return dist

def HISNN_train(source, target, K=10):
    '''
     HISNN_TRAIN() Summary of this function goes here,

     # Detailed explanation goes here,
    :param source: a mat array, each row is an instance, the last column is the label {0, 1}.
    :param target: a mat array, each row is an instance and the last column is the label {0, 1}.
                  Note that the label of target will not be used in the process of model training.
    :param K: the number of nearest neighbors.
    :return: model
    '''
    target_x = target[:, : - 1]
    source_x = source[:, : - 1]
    # print('len(target):', len(target), ';len(source):', len(source))

    # log Normalization only used when using NB classifier  (clf_index == 0)
    source_x = Log_transformation(source_x, alpha=0.000001)
    target_x = Log_transformation(target_x, alpha=0.000001)
    # print('len(source_x):', len(source_x))

    # Identify outlier based on mahalanobis distance
    # step(1)-By using distribution of source
    # st = np.cov(source_x.T)
    ms_rank = np.linalg.matrix_rank(np.cov(source_x.T))
    # print('con:', ms_rank == source_x.shape[1])
    # print('det:', np.round(det(np.cov(target_x.T)), -12))
    # if det(np.cov(source_x.T)) != 0 :  # Compute the determinant of a matrix   and (ms_rank == source_x.shape[1])
    if np.abs(det(np.cov(source_x.T))) >= 10e-6:
    #     print("行列式:", det(np.cov(source_x.T)))
        sx_d = np.linalg.inv(np.cov(source_x.T))
        # print("逆矩阵为：", sx_d)
        # print("rank:", ms_rank)
        # # m = np.cov(source_x.T)
        # print("行数和列数：", source_x.shape[0], source_x.shape[1])
        Dsms = mahal(source_x, source_x)
        # print('Dsms1:', len(Dsms))  # Dsms,
    else:
        Dsms = pdist(np.vstack((np.mean(source_x, axis=0), source_x)), 'euclidean')   # 垂直拼接
        # print('Dsms2:', len(Dsms))    #Dsms 57630

    Dsms = Dsms[: len(source_x)]
    std1 = np.std(Dsms)
    idx_outlier1 = []
    for j in range(len(Dsms)):
        if Dsms[j] > 3 * std1:
            idx_outlier1.append(j)  #  find(vector) - Return a column vector denoting the index of all non - zero elements in vector.
    # print('idx_outlier1', idx_outlier1, len(idx_outlier1))

    # step(2)-By using distribution of target
    # np.linalg.inv()
    # tt = np.cov(target_x.T)
    # mt_rank = np.linalg.matrix_rank(np.cov(target_x.T))
    # print('con:', mt_rank == target_x.shape[1])
    # print('det:', np.round(det(np.cov(target_x.T)), -12))
    if det(np.cov(target_x.T)) != 0:  # and (mt_rank == target_x.shape[1])
        # Mahalanobis distance
        Dtms = mahal(source_x, target_x)
        # print('Dtms1:', len(Dtms))  # , Dtms
    else:
        # Euclidean distance
        Dtms = pdist(np.vstack((np.mean(target_x, axis=0), source_x)), 'euclidean')
        # print('Dtms2:', len(Dtms))  #  , Dtms

    Dtms = Dtms[: len(source_x)]
    std2 = np.std(Dtms)
    idx_outlier2 = []
    for l in range(len(Dtms)):
        if Dtms[l] > 3 * std2:
            idx_outlier2.append(l)
    # print('idx_outlier2:', idx_outlier2, len(idx_outlier2))

    if len(idx_outlier2) > 0.8 * len(source):
        # In some case, len(idx_outlier2) == len(source), if so, all source data is outlier, i.e., there is no training data.
        idx_outlier2 = []

    idx_outlier1.extend(idx_outlier2)
    idx_outlier = np.unique(idx_outlier1)
    # print('idx_outlier:', len(idx_outlier))  #  idx_outlier,
    index = list(range(len(source)))
    idx_inlier = list(set(index) - set(idx_outlier))  # inlier index

    #step(3)-  KNN Filter based on Hamming distance
    inds = []
    da = np.zeros((len(source_x), len(target_x)))  # initialize the distance from a testing instance to each source instance.
    for i in range(len(target_x)):
        for j in range(len(source_x)):   # each instance in source data
            # Calculate hamming distance between a target instance and a source instance
            da[j, i] = pdist(np.vstack((target_x[i, :], source_x[j, :])), 'hamming')
        ind = da[:, i].argsort().tolist()  # ascending order 数组升序排序返回索引，并数组转换为列表
        inds.extend(ind[: K])  # 选择距离最近的k个值的索引列表

    # print('inds:', len(np.unique(inds)))
    inds.extend(idx_inlier)  # len(inds+inlier)
    # print('len:', len(source), len(np.unique(inds)))
    new_source = source[np.unique(inds), :]  # Remove the duplicated instances
    # print('len(new_source):',  len(new_source))  # , new_source[:2, ]
    return new_source

def HISNN_test(model, target, source):
    '''
    HISNN_test() : Summary of this function goes here
    # Detailed explanation goes here
    :param model:
    :param target: 2D array,
    :param source: 2D array,
    :return: measures, dict
    '''
    target_x = target[:, :-1]
    source_x = source[:, :-1]
    source_label = source[:, -1]
    target_label = target[:, -1]
    # probPos = []
    probPos = np.zeros((len(target_x), 1))

    for i in range(len(target_x)):
        Dts = []  # Dts: distance between target and source
        for j in range(len(source_x)):
            # print(len(target_x), len(source_x))
            Dts_i = pdist(np.array([target_x[i, :], source_x[j, :]]), 'hamming')  # np.array([target_x[i, :], source_x[j, :]])
            # print('j:', j, 'Dts_i:', Dts_i)
            Dts.append(Dts_i.tolist()[0])
            #  min_Dts = min(Dts)
        min_ts_index = indexofMinMore(Dts)

        source_label_local = []
        for tsindex in min_ts_index:
            source_label_local.append(source_label[tsindex])

        if len(min_ts_index) == 1:    # Case1
            idx_nns = min_ts_index[0]
            nns = source_x[idx_nns, :]   # the nearest source instance
            # source_x[idx_nns, :] = []
            range_list = []  # set the nns as null
            if idx_nns == len(source_x) - 1:  # the final index of samples
                range_list.extend(list(range(len(source_x-1))))
            else:
                leftlist = list(range(idx_nns))
                leftlist.extend(list(range(idx_nns+1, len(source_x))))
                range_list.extend(leftlist)
            Dss = []  # Dss: distance between nearest source and the left source
            for k in range_list:
                Dss_k = pdist(np.array([nns, source_x[k, :]]), 'hamming')  # Dss: distance between target and source
                Dss.append(Dss_k.tolist()[0])

            min_ss_index = indexofMinMore(Dss)
            subsource_label_local = []
            for ssindex in min_ss_index:
                subsource_label_local.append(source_label[ssindex])

            if len(min_ss_index) == 1:  # Case1.1
                # print('case1.1')

                probPos_i = model.predict_proba(target_x[i, :].reshape(1, -1))
                if model.classes_[1] == 1:
                    # probPos.append(probPos_i[0][1])
                    probPos[i] = probPos_i[0][1]
                else:
                    # probPos.append(probPos_i[0][0])
                    probPos[i] = probPos_i[0][0]
            elif len(np.unique(subsource_label_local)) == 1:  # Case1.2
                if np.unique(subsource_label_local) == 1:
                    probPos[i] = 1
                    # probPos.append(1)
                else:
                    probPos[i] = 0
                    # probPos.append(0)
            else:  # Case1.3
                # probPos[i] = model.predict_proba(target_x[i, :].reshape(1, -1))
                probPos_i = model.predict_proba(target_x[i, :].reshape(1, -1))
                if model.classes_[1] == 1:
                    probPos[i] = probPos_i[0][1]
                    # probPos.append(probPos_i[0][1])
                else:
                    probPos[i] = probPos_i[0][0]
                    # probPos.append(probPos_i[0][0])


        elif len(np.unique(source_label_local)) == 1:  # Case2
            if np.unique(source_label_local) == 1:
                probPos[i] = 1
                # probPos.append(1)
            else:
                probPos[i] = 0
                # probPos.append(0)
        else:  # Case3
            # probPos[i] = model.predict_proba(target_x[i, :].reshape(1, -1))  # p - the probability of being positive
            probPos_i = model.predict_proba(target_x[i, :].reshape(1, -1))
            if model.classes_[1] == 1:
                probPos[i] = probPos_i[0][1]
                # probPos.append(probPos_i[0][1])
            else:
                probPos[i] = probPos_i[0][0]
                # probPos.append(probPos_i[0][0])

    # print("probPos:\n", probPos, len(probPos))
    predLabel = (np.array(probPos) >= 0.5).astype(np.int32)
    # print('predLabel:\n', predLabel, len(predLabel))
    measures = performance(target_label, probPos)   # Call Performance()
    return measures


def HISNN_main(mode, clf_index, runtimes):

    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # datasets for RQ2
    datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.4.csv', 'xerces-init.csv'],
            ['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv'],
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
                                = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug', r=r)  # random sample 90% negative samples and 90% positive samples
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
                    # *******************HISNN*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    source_HISNN = HISNN_train(source, target, K=10)

                    # /* step3: Model building and evaluation */
                    # Train model: LR as classifier, model requires the label must beong to {0, 1}.
                    modelname_hisnn, model_hisnn = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_hisnn)
                    # print("modelname_hisnn:", modelname_hisnn)
                    model_hisnn.fit(source_HISNN[:, : -1], source_HISNN[:, -1])  # build models
                    measures_hisnn = HISNN_test(model_hisnn, target, source)  # measures_hisnn = HISNN_test(model, target_log, source_log)  # evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_hisnn.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(source), 'test_len': len(target),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_hisnn,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_hisnn, index=[r])
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

    # X = np.random.randint(1, 10, (10, 3))
    # Y = X
    # # print(X)
    # dist = mahal(Y, X)
    # # print(dist)

    mode = ['null', 'M2O_CPDP', 'eg_HISNN']
    foldername = mode[1] + '_' + mode[2]
    # NB classifier needs to log normalization for HISNN
    HISNN_main(mode=mode, clf_index=0, runtimes=2)

