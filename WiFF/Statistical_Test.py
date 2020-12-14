#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division  # Ensure division returns float
import os
import copy
import time
import pandas as pd
import numpy as np  # version >= 1.7.1 && <= 1.9.1
import re
from pandas.io.common import EmptyDataError
# from minepy import MINE
from multiprocessing import Pool
import multiprocessing
import scipy.stats as stats
from statistics import mean, stdev  # version >= Python3.4
# import matplotlib.pyplot as plt


def Wilcoxon_rank_sum_test(data1, data2):
    '''
    在统计学中，Wilcoxon rank-sum test（威尔科克森秩和检验）,
    也叫 Mann-Whitney U test（曼-惠特尼 U 检验），或者 Wilcoxon-Mann-Whitney test。
    秩和检验是一个非参的假设检验方法，一般用来检测 2 个数据集是否来自于相同分布的总体。
    在秩和检验中，我们不要求被检验的 2 组数据包含相同个数的元素，
    换句话说，秩和检验更适用于非成对数据之间的差异性检测。
    假设:
       每个样本中的观察是独立同分布的（iid）。
       可以对每个样本中的观察进行排序。
    解释:
      原假设 H0：两个样本的分布相等。
      备选假设 H1：两个样本的分布不相等。
    目标：拒绝零假设，有统计显著α=5%, p<<0.05
    :param data1:
    :param data2:
    :return:
    '''
    stat, p = stats.mannwhitneyu(data1, data2)
    return stat, p

def Wilcoxon_signed_rank_test(data1, data2):
    '''
    威尔科克森符号秩检验（WILCOXON SIGNED-RANK TEST）:一种非参的假设检验方法，
    它成对的检查 2 个数据集中的数据（即 paired difference test）来判断 2 个数据集是否来自相同分布的总体。
    假设:
       每个样本中的观察是独立同分布的（iid）。
       可以对每个样本中的观察进行排序。
    解释:
       原假设 H0：两个样本的分布均等。
       备选假设 H1：两个样本的分布不均等。
    拒绝零假设，有统计显著α=5%, p<<0.05
    :param data1:
    :param data2:
    :return:
    '''
    stat, p = stats.wilcoxon(data1, data2)
    return stat, p

def Friedman_chi_square_test(data1, data2):
    '''
    FRIEDMAN检验（FRIEDMAN TEST）:检验两个或更多配对样本的分布是否相等。
    假设:
        每个样本中的观察是独立同分布的（iid）。
        可以对每个样本中的观察进行排序。
        每个样本的观察是成对的。
    解释:
        H0：所有样本的分布均等。
        H1：一个或多个样本的分布不均等。
    拒绝零假设，有统计显著α=5%, p<<0.05
    :param data1:
    :param data2:
    :return:
    '''
    stat, p = stats.friedmanchisquare(data1, data2)
    return stat, p

def Kruskal_wallis_H_test(data1, data2):
    '''
    KRUSKAL-WALLIS H检验（KRUSKAL-WALLIS H TEST）:检验两个或多个独立样本的分布是否相等。
    假设:
        每个样本中的观察是独立同分布的（iid）。
        可以对每个样本中的观察进行排序。
    解释:
        H0：所有样本的分布均等。
        H1：一个或多个样本的分布不均等。
    拒绝零假设，有统计显著α=5%, p<<0.05
    :param data1:
    :param data2:
    :return:
    '''
    stat, p = stats.kruskal(data1, data2)
    return stat, p

def CliffDeltaValues(data1, data2):
    '''

    :param data1:
    :param data2:
    :return:
    '''
    return True
def Delta2Cohd(data1, data2):
    # https://rdrr.io/cran/orddom/man/delta2cohd.html
    return True

def Cohd2Delta(data1, data2):
    # https://rdrr.io/cran/orddom/man/cohd2delta.html
    return True

def Cohens_d(data1, data2):
    '''
    效应量：差异指标Cohen's d
    https://blog.csdn.net/illfm/article/details/19917181
    https://cloud.tencent.com/developer/article/1634971
    :param data1:
    :param data2:
    :return:
    '''
    d = (mean(data1) - mean(data2)) / (np.sqrt((stdev(data1) ** 2 + stdev(data2) ** 2) / 2))

    return d


def Example_Cohens_d(x, y):
    '''
    效应量：差异指标 Calculate Cohens' d Values
    注意分母是集合标准差，通常只有当两组的总体标准差相等时才适用
    :param x:
    :param y:
    :return:
    '''
    # from numpy import std, mean, sqrt
    nx = len(x)
    ny = len(y)
    if len(x) == len(y):
        # correct only if nx=ny
        d1 = (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)
        print("d by the 1st method = " + str(d1))
        return d1
    else:
        # print("The first method is incorrect because nx is not equal to ny.")
        # correct if the population S.D. is expected to be equal for the two groups.
        # correct for more general case including nx !=ny
        dof = nx + ny - 2
        d2 = (np.mean(x) - np.mean(y)) / np.sqrt(
            ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
        print("d by the more general 2nd method = " + str(d2))
        return d2

def T_test_CI_R2_Cohensd(data, pop_mean=20, alpha=0.05, t_ci=2.62):
    '''
    T-test for one-sample test

    :param data: one-sample test, n*1 array
    :param pop_mean: default = 20
    :param alpha: type: float, cut-off value, default = 0.05
    :param t_ci:
    :return: t, p, p_oneTail, CI, d, R_2
    '''
    sample_mean = np.mean(data)
    sample_std = np.std(data)
    n = len(data)
    # # 手动计算
    # ####### step1: 标准误差 #########
    # se = sample_std / (np.sqrt(n))  #1 标准误差 = 样本标准差 / (n的开方)
    # print('标准误差', se)
    # ####### step2: 计算t值 #########
    # t = (sample_mean - pop_mean) / se   # 2计算t值,查表t对应的概率p
    # ######## step3: 查表t对应的概率 ######
    # print('t=', t)
    # print('双尾检验的p=', p)
    # p_oneTail = p / 2  # 单尾检验的值
    # print('单尾检验p=', p_oneTail)

    # 自动计算
    # ####### Calculate t and p #########
    t, p = stats.ttest_1samp(data, pop_mean)
    p_oneTail = p / 2
    if (t < 0 and p_oneTail < alpha):
        print('拒绝零假设')
    else:
        print('接受零假设')
    print('t=', t, '双尾检验的p=', p, '单尾检验p=', p_oneTail)
    #
    # t_ci = 2.262

    ####### Calculate 置信区间 CI ###########
    se = stats.sem(data)  # 使用scipy计算标准误差
    # 置信区间上限
    a = sample_mean - t_ci * se
    # 置信区间下限
    b = sample_mean + t_ci * se
    CI = [a, b]
    print('单个平均值的置信区间，95置信水平 CI=(%f,%f)' % (a, b))

    ####### #Calculate 效应量：差异指标 Cohen's d #########
    d = (sample_mean - pop_mean) / sample_std

    ####### #Calculate 效应量：相关度指标 R^2 #########
    df = n - 1  # 自由度
    R_2 = (t * t) / (t * t + df)

    return t, p, p_oneTail, CI, d, R_2

def WinTieLoss(p1, p2):
    '''

    :param data1:
    :param data2:
    :return:
    '''
    win, tie, loss
    win = 0
    tie = 0
    loss = 0
    return True

def StatTest(fpath, spath):
   '''

   :param fpath:
   :param spath:
   :return:
   '''
   global N  # the result number of one test data. For M2O, N = 30; For O2O, it depends.
   filelist = os.listdir(fpath)
   if not os.path.exists(spath):
       os.mkdir(spath)
   for f in filelist:
       if f.split("_")[0] == 'all':
           print('当前文件路径：', fpath + '\\' + f)
           csvfile = open(fpath + '\\' + f, encoding='utf-8')
           df = pd.read_csv(csvfile)  # 把文件中所有的指标值读出来，不读文件其他信息
           measures = df.columns.values[1:5]
           # print(df.columns.values)
           length = df.shape[0]
           width = df.shape[1]
           if 'M2O' in fpath.split('\\')[-1]:
               N = 30
           elif 'O2O' in fpath.split('\\')[-1]:
               if fpath.split('\\')[-2] == 'Relink':
                   N = 60
               elif 'CK' in fpath.split('\\')[-2]:
                   N = 390
               elif fpath.split('\\')[-2] == 'AEEEM':
                   N = 120
           for k in range(0, 4, 1):  # four measures
               df_eachdata = pd.DataFrame()
               for i in range(0, length, N):
                   pd_eachdata_list = []
                   for j in range(1, (width - 4), 4):   # the first column is 'testfile' not 'measures
                       data1 = df.iloc[i:i + N, -4 + k]
                       data2 = df.iloc[i:i + N, j + k]
                       stat, p = Wilcoxon_signed_rank_test(data1, data2)
                       d = Cohens_d(data1, data2)
                       pd_eachdata_list.append(p)
                       pd_eachdata_list.append(d)
                   # print(pd_eachdata_list)
                   df_eachdata = pd.concat([df_eachdata, pd.DataFrame(pd_eachdata_list).T], axis=0)
               index = df['testfile'].drop_duplicates()
               # print(len(df_eachdata), index, '\n')
               df_eachdata.index = index
               if len(df_eachdata.columns.values) == 8:
                   df_eachdata.columns = ['CPDP-iForest_p', 'CPDP-iForest_d', 'CPDP-pre_p', 'CPDP-pre_d',
                                          'CPDP-SMOTE_p', 'CPDP-SMOTE_d', 'CPDP-WIFF_p', 'CPDP-WIFF_d']

               elif len(df_eachdata.columns.values) == 10:  # because of TNB without LR & RF
                   df_eachdata.columns = ['Bellwether_p', 'Bellwether_d', 'BF_p', 'BF_d',
                                          'DFAC_p', 'DFAC_d', 'HISNN_p', 'HISNN_d',
                                          'HBSF_p', 'HBSF_d']

               elif len(df_eachdata.columns.values) == 12:
                   df_eachdata.columns = ['Bellwether_p', 'Bellwether_d', 'BF_p', 'BF_d',
                                      'DFAC_p', 'DFAC_d','HISNN_p', 'HISNN_d',
                                      'TNB_p', 'TNB_d', 'HBSF_p', 'HBSF_d']
               else:
                   print('columns is wrong, please check')
               # print('On one dataset, the p, d values:', df_eachdata)
               savename = fpath.split('\\')[-1].split('_')[-1] + '_' \
                          + f.rstrip('.csv').split('_')[-1] + '_' + measures[k] + '_pd.csv'

               df_eachdata.to_csv(spath + '\\' + savename)



if __name__ == "__main__":
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # starttime = time.clock()  # 计时开始

    print("parent process %s" % (os.getpid()))
    pool = multiprocessing.Pool(8)
    #
    # data1 = [1, 2, 3.3, 5, 6]
    # data2 = [1, 3.2, 5, 6, 9.1]
    # data3 = [1, 3.2, 5, 6, 9.1, 4.6]
    # d = Cohens_d(data1, data2)  # d = d1
    # d1 = Example_Cohens_d(data1, data2)
    # d2 = Example_Cohens_d(data1, data3)
    # print(d, d1, d2)

    # test StatTest
    # fp = r'D:\PycharmProjects\cuican\data2020\ACR\AEEEM\concact_M2O'
    # sp = r'D:\PycharmProjects\cuican\data2020\ACR\AEEEM'
    # StatTest(fp, sp)


    # fps = [r'D:\PycharmProjects\cuican\data2020\ACR\AEEEM\concact_M2O',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\AEEEM\concact_O2O',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\Relink\concact_M2O',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\Relink\concact_O2O',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\CK\concact_M2O',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\CK\concact_O2O',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\CK2\concact_M2O',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\CK2\concact_O2O']
    #
    # sps = [r'D:\PycharmProjects\cuican\data2020\ACR\AEEEM',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\AEEEM',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\Relink',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\Relink',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\CK',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\CK',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\CK2',
    #        r'D:\PycharmProjects\cuican\data2020\ACR\CK2']
    # for i in range(len(fps)):
    #     StatTest(fps[i], sps[i])

    StatTest(r'D:\PycharmProjects\cuican\data2020\ACR\CK_RQ1\concact_O2O',
             r'D:\PycharmProjects\cuican\data2020\ACR\CK_RQ1')


