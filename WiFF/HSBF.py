#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
==============================================================================================================================
HSBF:Refer:(2018)Evaluating Data Filter on Cross-Project Defect Prediction: Comparison and Improvements
Main structure：
(1) select a train project by cosine;
(2) select instances by k-means from the selected project in step(1).
==============================================================================================================================


====================================================

@author: Cuican
@Affilation：Beihang University
@Date:2019/04/23
@email：cuican1414@buaa.edu.cn or cuic666@gmail.com

====================================================
"""

# print(__doc__)

import os
import sys
# sys.path.append(r"D:\PycharmProjects\data_preprocessing\CK_handle")
# sys.path.append(r'D:\PycharmProjects\data_preprocessing\CK_handle\compare_py')
import getopt
import copy
import time
from typing import Iterable

import pandas as pd
import numpy as np
import operator
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import sklearn.preprocessing as preprocessing

from datatransformation import DataCharacteristic, DataFrame2Array
from KMeansFilter import KMeansFilter
from DataProcessing import DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Process_Data, Random_Stratified_Sample_fraction, Log_transformation

from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model

def cosine_distance_vector(x, y):

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # bit_product_sum_xy = sum([item[0] * item[1] for item in zip(x, y)])
    # bit_product_sum_xx = sum([item[0] * item[1] for item in zip(x, x)])
    # bit_product_sum_yy = sum([item[0] * item[1] for item in zip(y, y)])
    # cos = bit_product_sum_xy / (np.sqrt(bit_product_sum_xx) * np.sqrt(bit_product_sum_yy))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))
    return cos

def cosine_distance_matrix(matrix1,matrix2):
    # method 4
    cos = cosine_similarity(matrix1, matrix2)

    # method 5
    # matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())
    # matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    # matrix1_norm = matrix1_norm[:, np.newaxis]
    # matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    # matrix2_norm = matrix2_norm[:, np.newaxis]
    # cos = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))

    return cos


def HSBF(trfilelist, tefile, k1, k2, r):
    '''
     cosine for selecting a similar project，kmeans for selecting instances or modules
    :param trfilelist: 绝对路径列表，列表中为字符串类型
    :param tefile: 绝对路径，字符串类型
    :param k1: default: k1=30
    :param k2: default: k2=10
    :param r: K-Means中的随机状态为r+1
    :return:
    '''
    t1 = time.time()
    time_dicts = {}
    global X_con, y_con  # X, y,
    trfile_select = []  # 初始化选择相似的源项目列表
    distancelist = []  # 初始化项目与项目的距离列表
    df_te_dict, Charac_te = DataCharacteristic(tefile)
    for trfile in trfilelist:
        if trfile != tefile:
            df_tr_dict, Charac_tr = DataCharacteristic(trfile)
            C_ori_tr = Charac_tr[:4, :-1]
            C_ori_te = Charac_te[:4, :-1]
            # print(C_ori_tr, C_ori_te)

            # 计算cosine距离
            shape_cte = C_ori_te.shape
            shape_ctr = C_ori_tr.shape
            # cos = cosine_distance_vector(C_ori_te, C_ori_tr)  # 向量的余弦值
            cos = cosine_distance_matrix(C_ori_te, C_ori_tr)   # 矩阵的余弦值
            cos_mean = np.mean(cos)  # 多维cosine值求平均
            cos_norm = 0.5 * cos_mean + 0.5  # 将【-1，1】之间归一化到【0，1】之间
            distancelist.append(cos_norm)  # 所有的欧氏距离列表

    sortedDistIndicies = np.array(distancelist).argsort().tolist()  # 排序并返回index
    selectlist = sortedDistIndicies[: k1]  # 选择距离最近的k个值的索引列表
    for i in range(len(selectlist)):
        trfile_select.append(trfilelist[selectlist[i]])

    X_te, y_te = DataFrame2Array(tefile)
    for trfile_sel in trfile_select:
        X_tr_sel_each, y_tr_sel_each = DataFrame2Array(trfile_sel)
        X_con = np.vstack((X_te, X_tr_sel_each))  # 垂直拼接
        y_con = np.hstack((y_te, y_tr_sel_each))  # 垂直拼接
    length_X_con = X_con.shape[0]
    length_te = X_te.shape[0]
    length_tr = length_X_con - length_te
    X_tr_ori = X_con[length_te:]
    y_tr_ori = y_con[length_te:]
    instance_time, X_tr_new, y_tr_new = KMeansFilter(X_tr_ori, y_tr_ori, X_te, y_te, k2, r)

    t2 = time.time()
    time_each = {'HSBF': (t2 - t1)}
    time_dicts.update(time_each)
    df_filter_time = pd.Series(time_dicts)

    return df_filter_time, X_tr_new, y_tr_new


def HSBF_main(mode, clf_index, runtimes):

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
                    trfilelist = []
                    for file_tr in datasets[i]:
                        if file_tr != file_te:
                            # print('train_file:', file_tr)
                            Address_tr = datapath + file_tr
                            trfilelist.append(Address_tr)
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
                    data_train_unique = Drop_Duplicate_Samples(pd.DataFrame(Sample_tr))  # drop duplicate samples
                    source = data_train_unique.values
                    target = np.c_[X_test, y_test]

                    # *******************HSBF*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    df_filter_time, X_train_new, y_train_new, = HSBF(trfilelist, Address_te, k1=10, k2=20, r=r)
                    y_train_new = preprocessing.Binarizer(threshold=0).transform(y_train_new.reshape(-1, 1))  #
                    # Train model: classifier / model requires the label must beong to {0, 1}.
                    modelname_hsbf, model_hsbf = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_hsbf)
                    # print("modelname:", modelname)
                    measures_hsbf = Build_Evaluation_Classification_Model(model_hsbf, X_train_new, y_train_new, X_test, y_test)  # build and evaluate models

                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_hsbf.update(
                        {'train_len_before': len(Sample_tr), 'train_len_after': len(Sample_tr), 'test_len': len(target),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_hsbf,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_hsbf, index=[r])
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


    mode = [[1, 1, 0, 'Weight_iForest', 'zscore', 'smote'], 'M2O_CPDP', 'eg_HSBF']
    foldername = mode[1] + '_' + mode[2]

    HSBF_main(mode=mode, clf_index=0, runtimes=2)