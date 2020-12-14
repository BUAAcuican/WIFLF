#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
================================================================================
Refer:(2017) data filtering method based on agglomerative clustering (DFAC)
Main structure：

================================================================================


==========================================

@author: Cuican
@Affilation：Beihang University
@Date:2019/04/23
@email：cuican1414@buaa.edu.cn or cuic666@gmail.com

==========================================
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
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans, AgglomerativeClustering

from DataProcessing import DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Process_Data, Random_Stratified_Sample_fraction, Log_transformation

from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model


def DFAC(X_tr, y_tr, X_te, y_te):
    '''
    【P195】【P194】（2017）data filtering method based on agglomerative clustering (DFAC)
    :return:
    '''
    time_dicts = {}
    t1 = time.time()

    # step1：训练集度量元数据和测试集度量元数据合并
    X = np.vstack((X_te, X_tr))  # 垂直拼接
    y = np.hstack((y_te, y_tr))
    length_X = X.shape[0]
    length_te = X_te.shape[0]
    length_tr = length_X - length_te
    # X_tr_ori = X[length_te:]
    # y_tr_ori = y[length_te:]

    # step2：聚类分簇
    c = int(np.floor(length_X / 10))
    # c = 10
    AC = AgglomerativeClustering(n_clusters=c, affinity='euclidean', memory=None,
                                 connectivity=None, compute_full_tree='auto', linkage='average',
                                 pooling_func='deprecated')  # 默认c=2
    AC.fit(X)
    labels = AC.labels_
    diff_labels = np.unique(labels)
    # print(labels, diff_labels)
    data = np.insert(X, -1, values=labels, axis=1)  # 末尾插入一列，为合并数据集添加簇标签
    length_data = data.shape[0]
    if length_data == length_X and data.shape[1] == X.shape[1] + 1:  # 判断合并数据是否正确
        pass
    else:
        print('数据合并报错')

    # step3：判断测试实例是否存在于簇中？存在，将训练集选出合并成训练集；不存在，删除没有测试实例存在的簇。
    del_index = []
    del_label = []
    retain_label = []
    retain_index = []
    for label in diff_labels:  # 注意len(labels) == X.shape[0], 因此必须设置为簇号列表，且簇号唯一
        index = np.where(data[:, -1] == label)[0].tolist()  # 簇号为label的样本索引(turple取值后为数组，再转为列表)
        # print(label, '\n', index)  # 检查簇及对应索引值
        index_test_instance = filter(lambda x: x < length_te, index)  # 筛选簇中索引为测试实例，返回filter
        index_test_instance = list(index_test_instance)  # 必须转为list才能使用, 簇中测试实例的索引列表
        index_train_instance = list(filter(lambda x: x >= length_te, index))

        if index_test_instance == list():
            # print('簇号为%d的数据无测试数据，删除该簇' % (label))
            del_index.extend(index)  # 列表插入数据，而不是列表和列表组合
            del_label.append(label)
        else:
            retain_label.append(label)
            retain_index.extend(index)
            # print('簇号为%d的数据中存在测试数据，因此,保留该簇' % (label))
    retain_label = np.unique(retain_label)  # 簇号去重
    del_label = set(del_label)  # 簇号去重
    # print('AC聚类删除的簇号为%s，保留的簇号为%s。\n' % (del_label, retain_label))
    X_new = np.delete(X, del_index, axis=0)  # 删除某些行
    y = np.ravel(y)
    y_new = np.delete(y, del_index, axis=0)
    X_tr_new = X_new[length_te:]
    y_tr_new = y_new[length_te:]

    if X_tr_new.shape[0] == y_tr_new.shape[0] \
            and X_tr_new.shape[0] + len(del_index) == X[length_te:].shape[0]:
        t2 = time.time()
        time_each = {str(DFAC): (t2 - t1)}
        time_dicts.update(time_each)
        df_filter_time = pd.Series(time_dicts)
    else:
        # print('删除训练样本出错，请检查！')
        pass

    # print('每个文件删除的样本行索引列表为 %s' % del_index)
    return df_filter_time, X_tr_new, y_tr_new

def DFAC_main(mode, clf_index, runtimes):

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

                    # random sample 90% negative samples and 90% positive samples
                    string = 'bug'
                    Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string, r=r)
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train_unique = Drop_Duplicate_Samples(pd.DataFrame(Sample_tr))  # drop duplicate samples
                    source = data_train_unique.values
                    target = np.c_[X_test, y_test]

                    # *******************DFAC*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train = source[:, :-1]
                    y_train = source[:, -1]
                    # only NB classifier
                    # X_train = Log_transformation(X_train)
                    # X_test = Log_transformation(X_test)
                    df_filter_time, X_train_new, y_train_new = DFAC(X_train, y_train, X_test, y_test)
                    # Train model: classifier / model requires the label must beong to {0, 1}.
                    modelname_dfac, model_dfac = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_dfac)
                    # print("modelname_dfac:", modelname_dfac)
                    measures_dfac = Build_Evaluation_Classification_Model(model_dfac, X_train_new, y_train_new, X_test, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_dfac.update(
                        {'train_len_before': len(Sample_tr), 'train_len_after': len(Sample_tr), 'test_len': len(target),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_dfac,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_dfac, index=[r])
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

    mode = [[1, 1, 0, 'Weight_iForest', 'zscore', 'smote'], 'M2O_CPDP', 'eg_DFAC']
    foldername = mode[1] + '_' + mode[2]
    DFAC_main(mode=mode, clf_index=0, runtimes=2)