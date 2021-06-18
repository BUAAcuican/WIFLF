#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
==========================================
Main
==========================================
@author: Cuican
@Affilation：Beihang University
@Date:2019/04/23
@email：cuican1414@buaa.edu.cn or cuic666@gmail.com

==========================================
程序的主结构：

==========================================
"""

# print(__doc__)

import os
import sys
sys.path.append(r"D:\PycharmProjects\data_preprocessing\CK_handle")
sys.path.append(r'D:\PycharmProjects\data_preprocessing\CK_handle\compare_py')
import getopt
import copy
import time
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans, AgglomerativeClustering







def PeterDiffFilter(X_tr, y_tr, X_te, y_te, k):
    '''
    P30 Peter Filter 2013; P174 Peng He rTDS 2014的复现
    使用K-means，以训练实例为驱动，为每个簇中的每个训练实例找到同簇最近点测试实例；
    使用KNN,为选中的测试实例从同簇中训练数据中选出k个最近邻点（相似点），
    即选出与测试集相似的去重源数据实例（2012）。
    :param X_tr: 训练样本的度量元数据
    :param y_tr: 训练样本的标签数据
    :param X_te: 测试样本的度量元数据
    :param y_te: 测试样本的标签数据
    :param k:  KNN中选择的最邻近点的数量，原文中k=1，因为簇设置的为np.floor(length_X/10)，保证每个簇中有十个混合样本；
               如果选择的最邻近点训练样本已经被选择，则选择第二个最近邻点
    :return: 样本选择的时间pd.Series，有去重的相似度训练度量元数据和标签数据
    '''
    time_dicts = {}
    # global df_filter_time
    t1 = time.time()
    r = 0

    # step1：训练集度量元数据和测试集度量元数据合并
    X = np.vstack((X_tr, X_te))  # 垂直拼接
    # X = np.concatenate((X_tr, X_te), axis=0)  # 垂直拼接
    # X = np.row_stack((X_tr, X_te))  # 垂直拼接
    length_X = X.shape[0]
    length_tr = X_tr.shape[0]
    length_te = X_te.shape[0]

    # step2：对合并数据集进行聚类(每个簇中有10个TDS+Testing)
    c = int(np.floor(length_X/10))
    km_clf = KMeans(n_clusters=c, init='k-means++', n_init=10, max_iter=300,
                    tol=1e-4, precompute_distances='auto', verbose=0,
                    random_state=r + 1, copy_x=True, n_jobs=None, algorithm='auto')  # , randnum
    km_clf.fit_transform(X)
    labels = km_clf.labels_
    data = np.insert(X, -1, values=labels, axis=1)  # 末尾插入一列，为合并数据集添加簇标签
    length_data = data.shape[0]
    if length_data == length_X and data.shape[1] == X.shape[1] + 1:  # 判断合并数据是否正确
        pass
    else:
        print('数据合并报错')

    # step3：判断测试实例是否存在于簇中？存在，将训练集选出合并成训练集；不存在，删除没有测试实例存在的簇。
    for label in set(labels):  # 注意len(labels) == X.shape[0], 因此必须设置为簇号列表，且簇号唯一
        index = np.where(data[:, -1] == label)[0].tolist()  # 簇号为label的样本索引(turple取值后为数组，再转为列表)
        # print(label, '\n', index)  # 检查簇及对应索引值
        index_test_instance = filter(lambda x: x >= length_tr, index)  # 筛选簇中索引为测试实例，返回filter
        index_test_instance = list(index_test_instance)  # 必须转为list才能使用, 簇中测试实例的索引列表
        index_train_instance = list(filter(lambda x: x < length_tr, index))

        if index_test_instance == list():
            print('簇号为%d的数据无测试数据，删除该簇' % (label))
        else:
            print('簇号为%d的数据中存在测试数据，因此,保留该簇' % (label))

            # step4:在保留的簇中，对于每个训练实例，找到同簇中的最近邻测试实例
            # (与X[index_train_instance, :]或data[index_train_instance,:-1]等价)
            cluster_X_tr = X_tr[index_train_instance, :]  # 簇中包含所有的训练实例
            cluster_y_tr = y_tr[index_train_instance]
            cluster_X_te = data[index_test_instance, :-1]  # 簇中包含所有的测试实例
            test_instance_true = index_test_instance-np.tile(length_tr, (1, len(index_test_instance)))  # 数组而不是列表
            cluster_y_te = y_te[test_instance_true.tolist()]  # 必须转换成列表索引，否则cluster_y_te为数组嵌套，索引报错
            # cluster_X = X[index, :]  # 簇中包含所有的实例
            time_tests, te_instances, te_labels = BurakFilter(cluster_X_te, cluster_y_te,
                                                              cluster_X_tr, cluster_y_tr, 1)  # 最近邻测试实例

            # step5:得到测试实例后，使用欧式距离找到同簇中源项目训练数据中的k个最近邻(k=1)
            time_trains, selectinstances, selectlabels = BurakDiffFilter(cluster_X_tr, cluster_y_tr,
                                                                         te_instances, te_labels, k)

            # step6:有去重地将这些近邻点组成一个新的训练集
            X_tr = np.concatenate((X_tr, selectinstances), axis=0)  # 垂直拼接
            y_tr = np.concatenate((y_tr, selectlabels), axis=0)

    X_tr_new = X_tr[length_tr:, :]
    y_tr_new = y_tr[length_tr:]

    t2 = time.time()
    time_each = {str(k): (t2 - t1)}
    time_dicts.update(time_each)
    df_filter_time = pd.Series(time_dicts)

    return df_filter_time, X_tr_new, y_tr_new

