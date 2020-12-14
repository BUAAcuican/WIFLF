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
    :return:esl: effect size level
    '''
    d = (mean(data1) - mean(data2)) / (np.sqrt((stdev(data1) ** 2 + stdev(data2) ** 2) / 2))
    esl = 'None'
    if np.abs(d) >= 0.8:
        esl = 'Large(L)'
    elif np.abs(d) < 0.8 and np.abs(d) >= 0.5:
        esl = 'Medium(M)'
    elif np.abs(d) < 0.5 and np.abs(d) >= 0.2:
        esl = 'Small(S)'
    elif np.abs(d) < 0.2:
        esl = 'Negligible(N)'
    else:
        print('d is wrong, please check...')

    return d, esl


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

def WinTieLoss(data1, data2, alpha, r):
    '''

    :param data1:
    :param data2:
    :param data1:
    :param data2:
    :return: wtl
    '''
    tie = 0
    win = 0
    loss = 0
    for i in range(0, len(data1), r):
        d1 = data1[i: i+r]
        d2 = data2[i: i+r]
        stat, p = Wilcoxon_signed_rank_test(d1, d2)
        d, esl = Cohens_d(d1, d2)
        if p > alpha:
            tie = tie + 1
        elif p <= alpha and d > 0:
            win = win + 1
        else:
            loss = loss + 1

    wtl = str(int(win)) + '/' + str(int(tie)) + '/' + str(int(loss))

    return wtl

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


def Average_Measures(fpath, spath, string):
    """
    Average_Measures() details is here：
    1) Fill Nan values as 0;
    2) Calculate mean/std/median of each dataset and all the datasets.
    :param fpath: absolute path of all measures of different approaches, type: string
                  In fpath, there are m * c csvfiles, m means No. of approaches, c means No. of classifiers;
                  A file contains all measure results of all datasets for run n times.
                  measures are: [AUC	Balance	MCC	G-Measure ...TP	TN	FP	FN  train_len_before
                                train_len_after	test_len runtime clf testfile trainfile runtimes]
                                (testfile and trainfile are string types, others are numeric.)
                  k datasets and n run times, o2o means one is as test data, the rest are as train data, respectively.
                  That is, there are k*(k-1)*n rows values and 1 row is column name.
    :param spath: absolute path of all fillNan/mean/median/STD of different approaches, type: string
                  fillNan files are just fill Nan of files in fpath as 0.
                  mean/median/STD files: A file contains k+1 *(len(measures)-1). The file lists the measures mean/std/median
                  value of every dataset and all dataset by one approach with one classifier.
    :return: Null
    """

    file_list = os.listdir(fpath)
    if not os.path.exists(spath):
            os.mkdir(spath)
    for file in file_list:
        data = pd.read_csv(fpath + '\\' + file, encoding='gbk')
        data_fillnan = data.fillna(0)

        savepath = spath + '\\' + fpath.split('\\')[-1] + '_results'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        data_fillnan.to_csv(savepath + '\\' + file, index=False)
        data_fillnan.drop(columns=['Unnamed: 0', 'runtimes'], inplace=True)
        # print(data_new.columns.values)
        gbr = data_fillnan.groupby(string, group_keys=0)  # Grouping criteria are not shown  # as_index=False,
        namelist = []
        df_mean = pd.DataFrame()
        df_std = pd.DataFrame()
        df_median = pd.DataFrame()
        for name, group in gbr:
            if '.csv' in name:
                namelist.append(name.rstrip('.csv'))
            else:
                namelist.append(name)
            # mean
            meaures_mean = group.mean()
            df_meaures_mean = pd.DataFrame(meaures_mean)
            df_meaures_mean = df_meaures_mean.T
            df_mean = pd.concat([df_mean, df_meaures_mean])
            # print(df_mean, type(df_mean))

            # std
            meaures_std = group.std()
            df_meaures_std = pd.DataFrame(meaures_std)
            df_meaures_std = df_meaures_std.T
            df_std = pd.concat([df_std, df_meaures_std])

            meaures_median = group.median()
            df_meaures_median = pd.DataFrame(meaures_median)
            df_meaures_median = df_meaures_median.T
            df_median = pd.concat([df_median, df_meaures_median])

        # print(df_mean.tail(3), len(df_mean))

        df_mean = pd.concat([df_mean, pd.DataFrame(df_mean.mean()).T]) # mean of all datasets
        df_std = pd.concat([df_std, pd.DataFrame(df_mean.std()).T])  # std of different datasets
        df_median = pd.concat([df_median, pd.DataFrame(df_mean.median()).T])  # median of different datasets

        # print(df_mean.tail(2), len(df_mean))
        # print(namelist, type(namelist))
        df_mean.index = namelist + ['mean']
        df_std.index = namelist + ['std']
        df_median.index = namelist + ['median']


        df_mean.to_csv(savepath + '\\' + file.rstrip('.csv') + '_mean.csv')
        df_std.to_csv(savepath + '\\' + file.rstrip('.csv') + '_std.csv')
        df_median.to_csv(savepath + '\\' + file.rstrip('.csv') + '_median.csv')
        # print(df_mean.tail(3), len(df_mean))

def Concat_Average_Measures(fpath, spath, filestring, measurestring, name):
    '''
    concat average measures into one file
    :param fpath:
    :param spath:
    :param filestring:
    :param measurestring:
    :return:
    '''
    if not os.path.exists(spath):
            os.mkdir(spath)
    folderlist = os.listdir(fpath)
    # print(folderlist)
    df_concat = pd.DataFrame()
    for folder in folderlist:
        filelist = os.listdir(fpath + '\\' + folder)
        for file in filelist:
            if filestring in file:
                df = pd.read_csv(fpath + '\\' + folder + '\\' + file)
                df_measures = df[measurestring]
                df_measures.index = df.iloc[:, 0]
                df_measures.loc['method'] = file.rstrip('.csv')
                df_concat = pd.concat([df_concat, df_measures], axis=1, ignore_index=False)

    # df_concat.columns = filelist
    df_concat.to_csv(spath + '\\' + filestring + '_' + name + '.csv')

def Pvalues_Dvalues_ESL_WTL(fpath, spath, filestring, m, r):
    '''
    Calculate p-value by Wilcoxon_signed_rank_test(), d values and effect size level by Cohens_d(),
    win/tie/loss results by WinTieLoss().
    :param fpath: absolute path about files waiting for statistical test. All the files in 'fpath' are normal csv file.
                  i.e., ============================
                        |  A,|   B,|   C,|    D,|...
                        ============================
                        |0.5,|0.79,| 0.7,| 0.43,|...
                        ============================
    :param spath: absolute path about stats test results
    :param filestring: part save name of stats and wtl .csv files
    :param m: the index of column of data1
    :param r: the runtimes of one each dataset, i.e, the repeat values about one project
    :return: None
    '''
    if not os.path.exists(spath):
            os.mkdir(spath)
    filelist = os.listdir(fpath)
    for file in filelist:
        df = pd.read_csv(fpath + '\\' + file)
        wintieloss = []
        df_stas_all = pd.DataFrame()
        columns_wtl = []
        d1 = df.iloc[:, m].astype(float)   # keep float data
        for j in range(0, len(df.columns.values), 1):
            if df.columns.values[j] != df.columns.values[m]:
                d2 = df.iloc[:, j]  # keep float data
                alpha = 0.05
                wtl = WinTieLoss(d1, d2, alpha, r)
                columns_wtl.append(df.columns.values[j])
                pvalues = []
                dvalues = []
                effectsizelevel = []
                for i in range(0, len(d1), r):
                    data1 = df.iloc[i:i + r, -2].astype(float)  # keep float data
                    data2 = df.iloc[i:i + r, j].astype(float)  # keep float data

                    stas, p = Wilcoxon_signed_rank_test(data1, data2)
                    d, esl = Cohens_d(data1, data2)
                    pvalues.append(p)
                    dvalues.append(d)
                    effectsizelevel.append(esl)
                values = np.array([pvalues, dvalues, effectsizelevel]).T

                wintieloss.append(wtl)
                columns = [df.columns.values[j] + '_p',
                           df.columns.values[j] + "_d", df.columns.values[j] + "_esl"]
                df_stas = pd.DataFrame(values, columns=columns)

                df_stas_all = pd.concat([df_stas_all, df_stas], axis=1)

            df_stas_all.to_csv(spath + '\\' + 'stas_' + filestring + '.csv')

            # print(columns_wtl)
            df_wtl = pd.DataFrame(np.array(wintieloss), index=columns_wtl).T

            df_wtl.to_csv(spath + '\\' + 'wtl_' + filestring + '.csv')


def P_D_ESL_WTL(fpath, spath, filestring, r):
    '''
    specially for RQ1
    :param fpath:
    :param spath:
    :param filestring:
    :return:
    '''
    if not os.path.exists(spath):
            os.mkdir(spath)
    filelist = os.listdir(fpath)
    for file in filelist:
        if filestring in file:
            df0 = pd.read_csv(fpath + '\\' + file)
            method = df0.iloc[-1, :].values
            # method = np.delete(method, 0)
            # print(method)
            df = pd.read_csv(fpath + '\\' + file)
            df.columns = method
            df.drop(columns=['method'], inplace=True)
            # print(df.index, df.columns)
            # df = df.astype(float)
            # df.set_index(df, inplace=True)
            d1 = df.iloc[:-1, -2].astype(float)   # WiFFd

            wintieloss = []
            df_stas_all = pd.DataFrame()
            columns_wtl = []
            for j in range(0, len(df.columns.values)-2, 1):
                if 'CPDP_SMOTE_iFl_d' not in df.columns.values[j]: # CPDP_SMOTE_iFl_d_LR':
                    d2 = df.iloc[:-1, j].astype(float)  # keep float data
                    # print(d2.head(3))
                    wtl = WinTieLoss(d1, d2, alpha=0.05, r=30)
                    columns_wtl.append(df.columns.values[j])
                    pvalues = []
                    dvalues = []
                    effectsizelevel = []
                    for i in range(0, len(d1), r):
                        data1 = df.iloc[i:i + r, -2].astype(float)  # keep float data
                        data2 = df.iloc[i:i + r, j].astype(float)  # keep float data

                        stas, p = Wilcoxon_signed_rank_test(data1, data2)
                        d, esl = Cohens_d(data1, data2)
                        pvalues.append(p)
                        dvalues.append(d)
                        effectsizelevel.append(esl)
                    values = np.array([pvalues, dvalues, effectsizelevel]).T

                    wintieloss.append(wtl)
                    columns = [df.columns.values[j] + '_p',
                               df.columns.values[j] + "_d", df.columns.values[j] + "_esl"]
                    df_stas = pd.DataFrame(values, columns=columns)

                    df_stas_all = pd.concat([df_stas_all, df_stas], axis=1)



            df_stas_all.to_csv(spath + '\\' + 'stas_' + filestring + '.csv')

            # print(columns_wtl)
            df_wtl = pd.DataFrame(np.array(wintieloss), index=columns_wtl).T

            df_wtl.to_csv(spath + '\\' + 'wtl_' + filestring+ '.csv')

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

    #################for CK-M2O RQ1 #################
    pwd = os.getcwd()
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    spath_e = father_path + '/results/'
    spathe_mean = father_path + '/results/mean'
    if not os.path.exists(spathe_mean):
        os.mkdir(spathe_mean)
    fl = os.listdir(spath_e)
    # print(fl)
    string = 'testfile'
    for folder in fl:
        fpath = spath_e + '\\' + folder
        print(fpath)
        Average_Measures(fpath, spathe_mean, string)

    # spath_concat =  father_path + '/results/M2O_concat'
    # filestrings = ['RF_mean', 'LR_mean', 'LR_std', 'RF_std', 'LR_median', 'RF_median']
    # for filestring in filestrings:
    #     measurestring =['Balance', 'G-Measure', 'G-Mean2']
    #     Concat_Average_Measures(spathe_mean, spath_concat, filestring, measurestring, name='measures')

    # filestrings = ['RF.csv', 'LR.csv']
    # for filestring in filestrings:
    #     measurestrings = [['Balance'], ['G-Measure'], ['G-Mean2']]
    #     for measurestring in measurestrings:
    #         name = measurestring[0]
    #         Concat_Average_Measures(spathe_mean, spath_concat, filestring, measurestring, name)

    # spath_stats = father_path +'/M2O_stats'
    # # filestring = 'LR.csv_Balance'
    # filestrings = ['LR.csv_Balance', 'LR.csv_G-Measure', 'LR.csv_G-Mean2',
    #                'RF.csv_Balance', 'RF.csv_G-Measure', 'RF.csv_G-Mean2']
    # for filestring in filestrings:
    #     P_D_ESL_WTL(spath_concat, spath_stats, filestring, r=30)

    #     StatTest(fps[i], sps[i])

