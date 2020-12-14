
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
import time


import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

def KMeansFilter(X_tr, y_tr, X_te, y_te, k, r):
    '''

    :param X_tr:
    :param y_tr:
    :param X_te:
    :param y_te:
    :param r: k-means的随机状态r+1，不影响聚类结果，因此设置为0
    :return:
    '''

    time_dicts = {}
    global df_filter_time
    t1 = time.time()
    # r = 0
    # step1：训练集度量元数据和测试集度量元数据合并
    length_tr = X_tr.shape[0]
    length_te = X_te.shape[0]
    y_te_type = type(y_te)
    X = np.vstack((X_tr, X_te))  # 垂直拼接
    # X = np.concatenate((X_tr, X_te), axis=0)  # 垂直拼接
    # X = np.row_stack((X_tr, X_te))  # 垂直拼接
    length_X = X.shape[0]

    # step2：对合并数据集进行聚类
    km_clf = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300,
                    tol=1e-4, precompute_distances='auto', verbose=0,
                    random_state=r + 1, copy_x=True, n_jobs=None, algorithm='auto')  # , randnum
    km_clf.fit_transform(X)
    centers = km_clf.cluster_centers_
    labels = km_clf.labels_
    data = np.insert(X, -1, values=labels, axis=1)  # 末尾插入一列，为合并数据集添加簇标签
    length_data = data.shape[0]
    if length_data == length_X and data.shape[1] == X.shape[1] + 1:  # 判断合并数据是否正确
        pass
    else:
        print('数据合并报错')

    # step3：判断测试实例是否存在于簇中？存在，将训练集选出合并成训练集；不存在，删除没有测试实例存在的簇。
    del_index = []  # 初始化要删除的索引列表
    del_label = []  # 初始化要删除的簇号列表
    retain_index = []  # 初始化要保留的索引列表
    retain_label = []  # 初始化要保留的簇号列表
    for label in set(labels):  # 注意len(labels) == X.shape[0], 因此必须设置为簇号列表，且簇号唯一
        index = np.where(data[:, -1] == label)[0].tolist()  # 簇号为label的样本索引(turple取值后为数组，再转为列表)
        # print(label, '\n', index)  # 检查簇及对应索引值
        index_test_instance = filter(lambda x: x >= length_tr, index)  # 筛选簇中索引为测试实例，返回filter
        index_test_instance = list(index_test_instance)  # 必须转为list才能使用, 簇中测试实例的索引列表
        if index_test_instance == list():
            # print('簇号为%d的数据无测试数据，删除该簇' % (label))
            del_index.extend(index)  # 列表插入数据，而不是列表和列表组合
            del_label.append(label)
        else:
            retain_label.append(label)
            retain_index.extend(index)
            # print('簇号为%d的数据中存在测试数据，因此,保留该簇' % (label))

    retain_label = set(retain_label)  # 簇号去重
    del_label = set(del_label)  # 簇号去重
    # print('第%d轮聚类，删除的簇号为%s，保留的簇号为%s。\n' % ((r + 1), del_label, retain_label))
    X_tr_new = np.delete(X_tr, del_index, axis=0)  # 删除某些行
    # X.drop(del_index)
    y_tr = np.ravel(y_tr)
    y_tr_new = np.delete(y_tr, del_index, axis=0)
    if X_tr_new.shape[0] == y_tr_new.shape[0] \
            and X_tr_new.shape[0] + len(del_index) == X_tr.shape[0]:
        t2 = time.time()
        time_each = {str(r): (t2 - t1)}
        time_dicts.update(time_each)
        df_filter_time = pd.Series(time_dicts)
    else:
        print('删除训练样本出错，请检查！')
    #
    # if del_index == []:
    #     print('每个文件中不删除样本')
    # else:
    #     print('每个文件删除的样本索引列表为 %s' % del_index)

    return df_filter_time, X_tr_new, y_tr_new