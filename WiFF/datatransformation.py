
#!/usr/bin/env Python
# coding=utf-8
"""
==========================================
Main
==========================================
@author: Cuican
@Affilation：Beihang University
@Date:2019/04/23
@email：cuican1414@buaa.edu.cn or cuic666@gmail.com
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
==========================================
程序的主结构：

==========================================
"""

# print(__doc__)

import os
import time
import pandas as pd
import numpy as np
import operator
from math import isnan

from collections import Counter

def classify(inX, dataSet, labels, k):
    '''

    :param inX: n×1维的样本点
    :param dataSet:
    :param labels:
    :param k:
    :return:
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 在行上重复dataSetSize次，列上重复1次
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5  # 计算欧式距离  sqrt(Σ(x-y)^2)
    sortedDistIndicies = distances.argsort()  # 排序并返回index

    # 选择距离最近的k个值
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # D.get(k[,d]) -> D[k] if k in D, else d. d defaults to None. 返回指定键的值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def DataCharacteristic(file):

    df = pd.read_csv(file)
    f_means = df.mean()
    f_maxs = df.max()
    f_mins = df.min()
    f_meadians = df.median()
    f_stds = df.std()
    Charac_dict = {'mean': f_means, "std": f_stds, 'min': f_mins, 'max': f_maxs, 'median': f_meadians}
    df_dict = pd.DataFrame.from_dict(Charac_dict, orient='index', columns=df.columns.values)
    Characteristic = df_dict.values
    return df_dict, Characteristic

def DataFrame2Array(file):
    """
    将pd.DataFrame类型转换为数组类型
    :param file:  文件名称（绝对路径）.
    :param X_te: A 2-D array that denotes test instances.
    :return: X——2-D array, type is float32;
             y——1-D array, type is float32;
    """
    df = pd.read_csv(file)  # 读取训练集(缺陷类)
    # df_tr['bug'].fillna(0.)
    col_name = df.columns.values
    label = df[col_name[-1]].unique()  # 查看标签的不同取值
    X = df[col_name[:-1]].values
    X.astype(np.float32)
    # print(X.shape, type(X))  # 查看数据类型，及样本量和特征数量
    y = df[col_name[-1]].values
    # y = y.reshape(-1, 1)  # 将一维向量转换为数组
    return X, y

def StatisticsCharacteristic(fpath):
    '''
    将所有的文件的数据分布特性合并为df.DataFrame
    :param fpath: 要读取的所有数据分布特性csv文件
    :return: 合并的df.DataFrame
    '''
    flist = os.listdir(fpath)
    dict_morefile = {}
    cloname = pd.read_csv(flist[0]).columns.values
    for f in flist:
        Charac_dict = DataCharacteristic(f)
        dict_morefile.update(Charac_dict)

    df = pd.DataFrame(data=dict_morefile, index=dict_morefile.keys(), columns=cloname)

    return df

def CheckNAN(string1, string2, s, path):
    '''
    文件夹中有多个子文件夹，子文件中有多个csv文件；或者，文件中没有子文件夹，有多个csv文件：
    判断文件中列名为AUC的列是否存在空值：
    如果存在空值，将空值所在行对于的列名为testfile的值保存至列表tename，并统计tename中各个元素出现的次数
    :param string1: 要判断的列名，字符串类型
    :param string2: 要判断的列名2，字符串类型
    :param s: 输出的文件名称，以‘_’划分的第s个字符，整数类型
    :param path: 要判断的绝对路径，字符串类型
    :return: 无
    '''
    global nante, nandic
    folderlist = os.listdir(path)
    for folder in folderlist:
        if not folder.endswith('.csv'):
            # if not os.path.isfile(folder):
            # if not os.path.isdir(folder):  # 判断是文件夹还是文件
            if not folder.endswith('.txt'):
                filelist = os.listdir(path + '\\' + folder)
                for file in filelist:
                    filename = path + '\\' + folder + '\\' + file
                    df = pd.read_csv(filename)
                    # print(df.columns)
                    tename = list()
                    for i in range(len(df)):

                        checkname = df[string1].iloc[i]
                        # Unamed:1
                        if isnan(checkname) is True:
                            # if checkname is not np.nan:  # #对于type 是float64 的空值，无法通过np.nan来过滤
                            test = df[string2].iloc[i]
                            tename.append(test)

                        else:
                            pass

                    print('文件夹和文件的名称分别为：', folder, '\n', file.rstrip('.csv').split('_')[s])
                    nante = np.unique(tename)  # 空值对应的测试文件
                    nandic = Counter(tename)

            else:
                print('文件为%s,已经没有可以读取的子文件！')
        else:
            filename = path + '\\' + folder
            df = pd.read_csv(filename)
            # print(df.columns)
            tename = list()
            for i in range(len(df)):
                checkname = df[string1].iloc[i]
                if isnan(checkname) is True:
                    # if checkname is not np.nan:  # #对于type 是float64 的空值，无法通过np.nan来过滤
                    test = df[string2].iloc[i]
                    tename.append(test)
                else:
                    pass
            nante = np.unique(tename)
            nandic = Counter(tename)
            print('文件所在路径及对应的文件名称：', path, '\n', folder.rstrip('.csv'))

        if len(nante) != int(0):
            print('空值对应的测试文件:', nante, '\n', '测试文件为空值的次数:', nandic)
        else:
            print('%s列没有空值！\n' % string1)

def ReadfileInfo(fpath,spath):
    fl = os.listdir(fpath)
    if not os.path.exists(spath):
        os.mkdir(spath)
    df_amb = pd.DataFrame()  # 歧义数据
    for f in fl:
        print('====step1: 统计原始数据的基本信息=====')
        df = pd.read_csv(fpath + '/' + f)
        column = df.columns.values  # 列名
        row = len(df)  # 行数df.shape[0]
        clo = df.shape[1]  # 列数
        df_X = df[column[:-1]]  # 读取数据的特征
        df_y = df[column[-1]]  # 读取数据的类标签
        # # 以下代码对字符类标签有效
        # bug_num = 0
        # bug_ori_num = 0
        # nbug_num = 0
        # for i in range(df.shape[0]):
        #     if df.iloc[i, -1] == 'buggy':
        #         bug_num += 1
        #         bug_ori_num += 1
        #     elif df.iloc[i, -1] == 'clean':
        #         nbug_num += 1
        #     else:
        #         print('类标签判断有误')

        # 下面几行代码针对数值类标签有效
        bug_ori_num = df_y.sum()  # 类标签的缺陷总数(类标签为数值型：缺陷数量而不是二元属性：0或1：无缺陷/有缺陷)
        # print(bug_num)
        mask_y = (df_y > 0)  # 判断类标签与0的大小,()由于优先级可以去掉
        bug_num = sum(mask_y)  # 有缺陷数量
        nbug_num = len(mask_y) - sum(mask_y)  # 无缺陷数量
        bug_r = np.around(bug_num / row, decimals=5)  # 缺陷模块占有率,四舍五入至小数点后5位
        print('%s文件中的基本信息如下：\n'
              '特征为：%d,样本量为:%d,\n '
              '类标签为二元时，正类为:%d个，负类为:%d个,\n'
              '类标签为数值型时，原始缺陷数量为：%d\n'
              '缺陷率为%.3f%%\n'
              % (f.rstrip('.csv'), clo - 1, row,
                 bug_num, nbug_num, bug_ori_num, bug_r * 100))


def InsertRepalceValues(fpath1, fpath2, spath):
    '''
    20190523
    存在文件名称相同的文件1和文件2，将文件1中某列==某值的所有行删除，
    并将删除后的剩余文件1‘和文件2合并成新的文件，命名不变，保存至新位置
    :param fpath1: 文件1所在文件夹的绝对路径
    :param fpath2: 文件2所在文件夹的绝对路径
    :param spath: 合并csv文件所在文件夹的绝对路径
    :return: 无
    '''
    orilist = os.listdir(fpath1)
    replist = os.listdir(fpath2)
    if not os.path.exists(spath):
        os.mkdir(spath)
    n = 0
    global df_new
    for repf in replist:
        for orif in orilist:
            if repf == orif:  # 如果两个文件的文件名称相同
                # 读取原始文件
                df_ori = pd.read_csv(fpath1 + '\\' + orif)
                # df_ori_rank = df_ori.groupby('testfile')  # 将原始结果按测试集名称排序
                df_ori_rank = df_ori.sort_values(by=["testfile", 'runtimes'], ascending=True)  # 升序
                # df_ori_rank.to_csv(spath + '\\' + repf, index=False)  # 将排序后的文件保存

                # 读取替换文件
                df_rep = pd.read_csv(fpath2 + '\\' + repf)
                testfname = df_rep['testfile'].values
                testfname2 = np.unique(testfname)  # 替换文件中的测试名称
                # print(len(df_ori_rank))
                del_index = []
                for i in testfname2:
                    old_values = df_ori_rank.loc[df_ori_rank['testfile'] == i]  # 某列等于某值的所有行
                    selec_index = df_ori_rank.index[df_ori_rank['testfile'] == i].tolist()   # 某列等于某值的所有行索引
                    del_index.extend(selec_index)
                df_del_rank = df_ori_rank.drop(del_index, axis=0)  # 删除含元素的所在行
                # print(len(df_ori_rank))

                # 判断删除是否正确，删除正确则合并文件；否则，报错
                del_num = len(testfname2) * 30   # 程序运行次数为30次
                if len(df_del_rank) + del_num == len(df_ori):
                    df_new = pd.concat([df_del_rank, df_rep])
                    if df_new['AUC'].isnull().any() == True:  # 检查是否仍然有空值存在
                        print('%s文件合并不成功' % orif)
                        df_new = df_new.sort_values(by=["testfile", 'runtimes'], ascending=True)  # 升序
                        df_new.to_csv(spath + '\\' + repf, index=False)
                    else:
                        # print('%s文件合并成功' % orif)
                        n += 1
                        df_new = df_new.sort_values(by=["testfile", 'runtimes'], ascending=True)  # 升序
                        df_new.to_csv(spath + '\\' + repf, index=False)
                else:
                    print('%s文件删除不成功' % orif)


    print('%s共合并%d个文件' % (fpath1, n))

def Checkrownumbers(path):
    flist = os.listdir(path)
    for f in flist:
        df = pd.read_csv(path + '\\' + f)
        testname = df['testfile'].values.tolist()
        testname2 = np.unique(testname)
        num = len(testname2) * 30
        if num == len(testname):
            print('right')
            pass
        else:
            print('%s is wrong' % f)

def DelNanValues(fpath1, fpath2):

    orilist = os.listdir(fpath1)
    replist = os.listdir(fpath2)
    n = 0
    global df_del_rank
    for orif in orilist:
        df_ori = pd.read_csv(fpath1 + '\\' + orif)
        # df_ori_rank = df_ori.groupby('testfile')  # 将原始结果按测试集名称排序
        df_ori_rank = df_ori.sort_values(by=["testfile", 'runtimes'], ascending=True)  # 升序

        for repf in replist:
            if repf == orif:
                df_rep = pd.read_csv(fpath2 + '\\' + repf)
                testfnames = df_rep['testfile'].values
                testfnames2 = np.unique(testfnames)  # 替换文件中的测试名称
                # print(len(df_ori_rank))
                del_index = []
                for tename in testfnames2:
                    selec_index = df_ori_rank.index[df_ori_rank['testfile'] == tename].tolist()  # 某列等于某值的所有行索引
                    del_index.extend(selec_index)
                df_del_rank = df_ori_rank.drop(del_index, axis=0)  # 删除含元素的所在行

                # print(len(df_ori_rank))
                del_num = len(testfnames2) * 30  # 程序运行次数为30次
                if len(df_del_rank) + del_num == len(df_ori):
                    pass
                else:
                    n += 1
                    print(len(df_del_rank), len(df_ori))
                    print('%s文件删除不成功' % orif)

    print('文件共%d个，删除不成功的有%d' % (len(orilist), n))


def DelSomeRowValues(fpath, valueslist, spath):

    # ant-1.3.csv、camel-1.0.csv、e-learning-1.csv、prop-6.csv、redaktor-1.csv
    orilist = os.listdir(fpath)

    n = 0
    global df_del_rank
    for orif in orilist:
        df_ori = pd.read_csv(fpath + '\\' + orif)
        del_index = []
        for tename in valueslist:
            selec_index = df_ori.index[df_ori['testfile'] == tename].tolist()  # 某列等于某值的所有行索引
            del_index.extend(selec_index)
        df_del_rank = df_ori.drop(del_index, axis=0)  # 删除含元素的所在行
        del_num = len(valueslist) * 30  # 程序运行次数为30次
        if len(df_del_rank) + del_num == len(df_ori):
            pass
        else:
            n += 1
            print(len(df_del_rank), len(df_ori))
            print('%s文件删除不成功' % orif)
        df_del_rank.to_csv(spath + '\\' + orif, index=None, header=True)


def SumRowTime(path, l, spath):
    '''
    间隔相加
    :param path:
    :param spath:
    :return:
    '''
    if not os.path.exists(spath):
        os.mkdir(spath)
    flist = os.listdir(path)
    for f in flist:
        df = pd.read_csv(path + '\\' + f)
        # print(df)
        df_time = df['0']

        timelist = []
        for i in range(0, len(df), l):
            df_sum = df_time.iloc[i: i+l].sum()
            print(i, '个相加的首值:', df_time.iloc[i])
            print("部分求和：", df_sum)
            timelist.append(df_sum)
        timeall = timelist * 30
        df_fitertime = pd.DataFrame(timeall)
        df_fitertime.to_csv(spath + '\\' + f.rstrip('.csv') + 'time.csv')

def SumColumnsTime(path, l, spath):
    '''
        间隔相加
        :param path:
        :param spath:
        :return:
        '''
    if not os.path.exists(spath):
        os.mkdir(spath)
    flist = os.listdir(path)
    df1 = pd.read_csv(path + '\\' + flist[0])
    df2 = pd.read_csv(path + '\\' + flist[1])
    df3 = pd.read_csv(path + '\\' + flist[2])
    df_Time = pd.DataFrame()
    for i in range(1, df1.shape[1], 1):
        df_sum_i = df1.iloc[:, i] + df2.iloc[:, i] + df3.iloc[:, i]

        df_fitertime.to_csv(spath + '\\' + f.rstrip('.csv') + 'time.csv')


def SortColumns(path, spath):
    '''
    按列名排序
    :param path:
    :param spath:
    :return:
    '''
    if not os.path.exists(spath):
        os.mkdir(spath)
    flist = os.listdir(path)

    fl = []
    for file in flist:
        if file.endswith('.csv'):
            fl.append(file)

    for f in fl:
        df = pd.read_csv(path + '\\' + f)
        df = df.sort_values(by=['runtimes', 'testfile'], ascending=True)  # 升序
        df.to_csv(spath + '\\' + f, index=False)


def ReplaceColumns(path, timepath, spath):
    '''
    替换某列的值
    :param path:
    :param spath:
    :return:
    '''
    if not os.path.exists(spath):
        os.mkdir(spath)
    flist = os.listdir(path)

    dftime = pd.read_csv(timepath + '\\' + os.listdir(timepath)[0])
    df_time = dftime['0']

    for f in flist:
        df = pd.read_csv(path + '\\' + f)
        df['filtertime'] = df_time
        print(df.head(3), df['filtertime'].head(3))
        df.to_csv(spath + '\\' + f, index=False)

def ReplaceColumns2(path, string, spath):
    '''
    读取某列的值，经过一些处理后，再替换该列的值
    :param path:
    :param spath:
    :return:
    '''
    if not os.path.exists(spath):
        os.mkdir(spath)
    flist = os.listdir(path)

    for f in flist:
        rep_values = []
        df = pd.read_csv(path + '\\' + f)
        ori_values = df[string].values.tolist()
        rep_values.extend([ori_values[15], ori_values[0], ori_values[15], ori_values[15]])
        rep_values.extend(ori_values[1:5])
        rep_values.extend([ori_values[15], ori_values[15]])
        rep_values.extend(ori_values[5:9])
        rep_values.extend([ori_values[15]])
        rep_values.extend(ori_values[15:18])
        df[string] = np.array(rep_values)

        print(df.head(3), df[string].head(3))
        df.to_csv(spath + '\\' + f, index=False)

def Concatfiles(fpath, name, spath):
    '''
    合并文件
    :param fpath:
    :param spath:
    :return:
    '''
    if not os.path.exists(spath):
        os.mkdir(spath)

    flist = os.listdir(fpath)
    df_m = pd.DataFrame()
    for f in flist:
        file = open(fpath + '\\' + f)
        df = pd.read_csv(file)
        df_m = pd.concat([df_m, df], sort=True)
        # if f.rstrip(f.split('_')[-1]) == tef.rstrip('.csv'):
    df_m.to_csv(spath + '\\' + name, index=None)
    print('%s文件对应的训练样本集已合并完成, 样本量为%s' % (fpath, len(df_m)))

def Concattwofiles(fpath, fpath2, spath):
    '''
        合并文件
        :param fpath:
        :param spath:
        :return:
        '''
    if not os.path.exists(spath):
        os.mkdir(spath)

    flist = os.listdir(fpath)
    flist2 = os.listdir(fpath2)

    for f in flist:
        for f2 in flist2:
            if f == f2:
                file1 = open(fpath + '\\' + f)
                file2 = open(fpath2 + '\\' + f2)
                df = pd.read_csv(file1)
                df2 = pd.read_csv(file2)
                df2 = pd.concat([df, df2], sort=False)
                df2.to_csv(spath + '\\' + f, index=None)
                print('%s文件对应的训练样本集已合并完成, 样本量为%s' % (fpath, len(df2)))

def Concattrainfiles(fpath, string, spath):
    '''
    合并文件
    :param fpath:
    :param spath:
    :return:
    '''
    if not os.path.exists(spath):
        os.mkdir(spath)

    flist = os.listdir(fpath)


    for i in range(len(flist)):
        df_m = pd.DataFrame()
        for j in range(len(flist)):
            if j != i:
                df = pd.read_csv(fpath + '\\' + flist[j])
                df_m = pd.concat([df_m, df])
        # if f.rstrip(f.split('_')[-1]) == tef.rstrip('.csv'):
        df_m.to_csv(spath + '\\' + string + flist[i], index=None)
        print('%s文件对应的训练样本集已合并完成, 样本量为%s' % (fpath, len(df_m)))

def Concatmorefiles(fpath, f, t, spath):

    global name
    fl = os.listdir(fpath)
    print(fpath, len(fl))
    df_m = pd.DataFrame()  # for i in range(f, t, 1):
    for i in range(len(fl)):
        str = fl[i].split('-')[-1]
        inter = int(str.rstrip('_NB.csv'))
        if inter < int(t) and inter >= int(f):
            j = i
            print(j)
            file = open(fpath + '\\' + fl[j])
            df = pd.read_csv(file)
            df_m = pd.concat([df_m, df])
    name = fl[2].rstrip(fl[2].split('-')[-1])
    print(name)
    df_m.to_csv(spath + '\\' + name.rstrip('-') + '_multi_' + '30' + '_NB.csv', index=None)
    print('%s文件对应的训练样本集已合并完成, 样本量为%s' % (fpath, len(df_m)))

def RenameMoreFiles(fpath, spath):
    fl = os.listdir(fpath)
    i = -1
    for f in fl:

        if "Results_TiFF" in f and not 'Results_TiFF-256' in f:
            i += 1
            print("原来的文件名字是:{}".format(f))
            # 找到老的文件所在的位置
            old_file = os.path.join(fpath, f)
            print("old_file is {}".format(old_file))
            # 指定新文件的位置，如果没有使用这个方法，则新文件名生成在本项目的目录中
            # new_file = os.path.join(spath, "Results_TiFF-256" + f.lstrip(f.split('-')[0]))  #

            new_file = os.path.join(spath, "Results_TiFF-256-0.01-" + str(i) + '_NB.csv')
            print("File will be renamed as:{}".format(new_file))
            os.rename(old_file, new_file)
            print("修改后的文件名是:{}".format(f))



def RankValues(fpath, STRING, num, spath):
    '''
    某行值按大小排序
    :param fp:
    :param sp:
    :param num:方法的数量，int
    :return:
    '''
    global columns, X
    if not os.path.exists(spath):
        os.mkdir(spath)
    flist = os.listdir(fpath)
    df_m = pd.DataFrame()
    X = np.zeros((len(flist), num))
    sort_index_list = []  # 按照数据从小到大对应的索引排序的列表
    rank_index_list = []
    for i in range(len(flist)):
        df = pd.read_csv(fpath + '\\' + flist[i])
        columns = df.columns.values
        mean = df.iloc[15, 1:]
        mean_values = mean.values
        df_m = pd.concat([df_m, mean])

        sort_index = np.argsort(mean_values)  # 按照从小到大的顺序，输出对应的索引值
        # print(sort_index)
        for j in range(len(sort_index)):
            X[i][sort_index[j]] = int(j)  # 按照原数据名次对应的索引排序的列表

        # rank_tuple = sorted(enumerate(mean_values.tolist()), key=lambda x: x[1])
        # rank_index_list.append(rank_tuple)  # 元组类型

    X_DF = pd.DataFrame(X, index=range(len(flist)), columns=columns[1:])
    print('每个过滤器对应的排名：\n', X_DF)
    print('每个过滤器的排名和(越大越好)：\n', X_DF.sum())
    X_DF.to_csv(spath + '\\' + 'release_mean_' + STRING +'_rank.csv', index=True, header=True)

    # sort = pd.DataFrame(np.array(sort_index_list), index=range(len(flist)), columns=columns[1:])
    # print('每个过滤器的排名和：', rank.sum())
    # sort.to_csv(spath + '\\' + 'release_mean_' + STRING +'_sort.csv', index=True, header=True)

def ConcatOneColumn(s1, s2, spath):
    fl1 = os.listdir(s1)
    file1list = []
    for f1 in fl1:
        if f1.endswith('.csv'):
            file1list.append(f1)

    fl2 = os.listdir(s2)
    file2list = []
    for f2 in fl2:
        if f2.endswith('.csv'):
            file2list.append(f2)

    for fpd1 in file1list:
        for fpd2 in file2list:
            if fpd1 == fpd2:
                df1 = pd.read_csv(s1 + '\\' + fpd1)
                df2 = pd.read_csv(s2 + '\\' + fpd2)
                df = pd.concat([df1, df2['HPF']], axis=1)
                df.to_csv(spath + '\\' + fpd2)

def MeanMoreValuesofOneColumn(s1, string, num, spath):
    if not os.path.exists(spath):
        os.mkdir(spath)

    fl1 = os.listdir(s1)
    file1list = []
    for f1 in fl1:
        if f1.endswith('.csv'):
            file1list.append(f1)
    means = pd.DataFrame()
    stds = pd.DataFrame()
    for f in file1list:
        df = pd.read_csv(s1 + '\\' + f)
        df.drop(['testfile', 'Unnamed: 0'], axis=1, inplace=True)
        sel_index = df.index[df[string] < num].tolist()  # 某列小于某值的所有行索引
        df_sel = df.iloc[sel_index]
        # del df_sel['Unnamed: 0']
        mean = df_sel.mean()
        std = df_sel.std()
        pd_mean = pd.DataFrame(mean)
        pd_std = pd.DataFrame(std)
        means = pd.concat([means, pd_mean.T], sort=True)
        stds = pd.concat([stds, pd_std.T], sort=True)
    means.index = file1list
    means.to_csv(spath + '\\' + 'mean_runtimes_' + str(num) + '.csv', index=True)
    stds.index = file1list
    stds.to_csv(spath + '\\' + 'std_runtimes_' + str(num) + '.csv', index=True)
    print(means, stds)

def RenameFiles(fpath, str1, str2):
    # 批量重命名或替换文件名称
    flist = os.listdir(fpath)  # 获得文件夹下文件名列表
    print(flist)
    # path = unicode(fpath, "utf8")
    os.chdir(fpath)  # 选择要重命名的文件夹路径
    for file in flist:
        os.rename(file, file.replace(str1, str2))  # 将文件名中的'-_multi_250_'用'_'字符串替代

def Checktestfile(fpath):

    flist = os.listdir(fpath)  # 获得文件夹下文件名列表
    # 文件的长度
    for f in flist:
        df = pd.read_csv(fpath + '\\' + f)
    print(f, '文件中数据数量有', len(df))
    tename = np.unique(df['testfile'].values.tolist())
    print(f, '文件中数据数量有', len(df), '目标集的数量和名称为', len(tename), tename)

    # for i in range(len(flist)):
    #     file = open(fpath + '\\' + flist[i])
    #     df = pd.read_csv(file, encoding='utf-8')
    #     if len(df)  != int(3750):
    #         print(flist[i], '文件中数据数量有', len(df))

def SplitOneColumn(fpath, n1, n2, spath):

    if not os.path.exists(spath):
        os.mkdir(spath)

    flist = os.listdir(fpath)  # 获得文件夹下文件名列表

    for f in flist:
        df = pd.read_csv(fpath + '\\' + f)
        fname = df['Unnamed: 0'].values.tolist()

        list1 = []
        list2 = []
        for l in fname:
            list1.append(l.split('_')[n1])  # 2
            list2.append(l.split('_')[n2])  # 3
        col_name = df.columns.tolist()
        col_name.insert(col_name.index('AUC'), 'subsample')  # 在 AUC 列前面插入
        col_name.insert(col_name.index('subsample') + 1, 'alpha')  # 在subsample 列后面插入
        col_name.insert(col_name.index('subsample'), 'r')
        # df.reindex(columns=col_name)
        df['subsample'] = list1
        df['alpha'] = list2
        rvalues = f.split('_')[2].rstrip('.csv')  # list(int())
        df['r'] = rvalues  # * len(df)
        print(df.head())
        df.to_csv(spath + '\\' + f, index=False)


def TransposeCSV(fpath, spath):
    if not os.path.exists(spath):
        os.mkdir(spath)

    filel = os.listdir(fpath)
    for f in filel:
        df = pd.read_csv(fpath + '\\' + f)
        df.T.to_csv(spath + '\\' + f, index=True, header=False)

if __name__ == "__main__":
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    fpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\aeeem_experiment\compareresults\20190711results\t'
    spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\aeeem_experiment\compareresults\20190711results\t_after'
    # TransposeCSV(fpath, spath)

    starttime = time.clock()  # 计时开始
    fp = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean\r=250\all'
    # df = pd.read_csv(fp + '\\' + 'Results_TiFF_50_0.02_NB.csv')
    # print(len(df))
    # Checktestfile(r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean\r=41，n=256')
    # sp1 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean\r=41，n=256'
    # sp2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean\r=250\256'
    # Concattwofiles(sp1, sp2, fp)

    sp3 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean\r30\r=30, n=256'
    sp4 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean\r=250\256'
    spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean\r=250\mean'
    f = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\parameters_discussions\n=256'
    # fl = os.listdir(f)
    # print(fl)
    # list = ['0.001', '0.01', '0.1', '0.02', '0.2', '0.03',
    #         '0.04', '0.005', '0.05', '0.5', '0.06', '0.07']
    # l = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\parameters_discussions'
    # for i in range(len(fl)):
    #     s = f + '\\' + fl[i]
    #     Concatfiles(s, 'Results_TiFF_256_' + list[i] + '_NB.csv', l)
    # num = [2, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200, 250]
    # for n in num:
    #     MeanMoreValuesofOneColumn(fp, 'runtimes', n, spath)
    spath2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean\r=250\handle'
    # SplitOneColumn(spath, 2, 3, spath2)
    mean = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean\r=250\mean_handle'
    cpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean\r=250'
    # Concatfiles(mean, 'all_para_result.csv', cpath)

    # ae0 = r'D:\PycharmProjects\data_preprocessing\20190717\rank_instance_release_AEEEM'
    # ck0 = r'D:\PycharmProjects\data_preprocessing\20190717\rank_instance_release_CK\rank_15datasets'
    # ae1 = r'D:\PycharmProjects\data_preprocessing\20190717\rank_instance_release_CK\rank_release'
    # ckae = r'D:\PycharmProjects\data_preprocessing\20190717\rank_20datasets'
    ae0 = r'D:\PycharmProjects\data_preprocessing\20190717\Mean_eachtestfile_AEEEM'
    ck0 = r'D:\PycharmProjects\data_preprocessing\20190717\Mean_eachtestfile_CK\CK_Mean_eachtestfile_15datasets'
    ae1 = r'D:\PycharmProjects\data_preprocessing\20190717\rank_instance_release_CK\rank_release'
    ckae = r'D:\PycharmProjects\data_preprocessing\20190717\Mean_eachtestfile_20datasets'
    # FL = os.listdir(ae1)
    # for file in FL:
    #     ae = ae1 + '\\' + file
    #     RenameFiles(ae, 'HeBurakFilter', 'HBF')
    #     RenameFiles(ae, 'HeFilter', 'HF')
    ae = r'D:\PycharmProjects\data_preprocessing\20190717\Mean_eachtestfile_AEEEM'
    RenameFiles(ae, 'GlobalFilter', 'GF')
    # RenameFiles(ae, 'HeBurakFilter', 'HBF')
    # RenameFiles(ae, 'HePeterFilter', 'HPF')
    # RenameFiles(ae, 'HerboldNNFilter_', 'HNNF_')
    Concattwofiles(ck0, ae0, ckae)

    # fpath1 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\tr'
    # fpath2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\te'
    #
    # df_dict, Characteristic = DataCharacteristic(fpath2 + '\\' + 'ant-1.3.csv')
    #
    # X_tr, y_tr = DataFrame2Array(fpath2 + '\\' + 'ant-1.3.csv')  # y_tr = y_tr.ravel()
    # X_te, y_te = DataFrame2Array(fpath2 + '\\' + 'arc-1.csv')
    # X_mtr, y_mtr = DataFrame2Array(fpath1 + '\\' + 'exp_ant-1.3.csv')
    # df = StatisticsCharacteristic(fpath1)
    #
    # tefilename = 'ant-1.3.csv'
    # tefile = fpath2 + '\\' + tefilename
    # filelist = os.listdir(fpath2)
    # trfilelist = []
    # for file in filelist:
    #     if not file == tefilename:
    #         trfilelist.append(fpath2 + '\\' + file)

    '''检查文件中指定列是否存在空值'''
    # path = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\NO_SMOTE'
    # path = r'D:\PycharmProjects\data_preprocessing\CK_handle\aeeem_experiment\compareresults'

    # path = r'D:\PycharmProjects\data_preprocessing\CK_handle\Relink_csv\compareresults'
    # path = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\completeresults'
    # path=r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\completeresults\HeBurakFilter'
    # path = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\NO_SMOTE\HeBurak'
    # CheckNAN('AUC', 'testfile', 2, path)

    # path = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_instance\CK_Mean_eachtef_allf_eachmr\select'
    #  BurakFilter	GlobalFilter	iForestFilterStr1	iForestFilterStr2	PeterFilter	WPCP
    # list1 = ['BurakFilter',	'GlobalFilter',	'iForestFilterStr1',
    #          'iForestFilterStr2', 'PeterFilter', 'WPCP']
    # for l in list1[1:]:
    #     CheckNAN(l, 'Unnamed: 0', 3, path)
    # CheckNAN(list1[0], 'Unnamed: 0', 3, path)

    # selectpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\CK_Mean_eachtef_allf_eachmr\select'
    # list2 = ['HeBurakFilter', 'HeFilter', 'HePeterFilter',
    #          'HerboldNNFilter', 'iForestFilter1Project', 'iForestFilter2Project', 'WPCP']
    # for l in list2[1:]:
    #     CheckNAN(l, 'Unnamed: 0', 3, selectpath)
    # CheckNAN(list2[5], 'Unnamed: 0', 3, selectpath)

    '''检查重新运行的程序是否正确：以运行列数正确为检查方式'''
    # path = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\buchong\replaceresults'
    # Checkrownumbers(path)

    '''第一个文件删除与第二个文件相同的列，然后将两个文件合并'''
    # Relink
    # fpath1 = r'D:\PycharmProjects\data_preprocessing\CK_handle\Relink_csv\compareresults\iForestFilterStr2'
    # fpath2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\Relink_csv\compareresults\replaceresults'
    # spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\Relink_csv\compareresults\concatresults'

    # AEEEM
    # fpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\Duptotaldata'
    # spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults'

    # # CK
    # fpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\NO_SMOTE'
    # fl = os.listdir(fpath)
    # usefulf = []
    # for fp in fl:
    #     if fp not in ['readme.txt']:
    #         usefulf.append(fp)
    # fpath2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\buchong\replaceresults'
    # spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\buchong\concatresults'
    # for f in usefulf:
    #     fpath1 = fpath + '\\' + f
    #     InsertRepalceValues(fpath1, fpath2, spath)
    #
    # f1 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\buchong\1'
    # f2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\buchong\2'
    # spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\buchong\concatresults'
    # InsertRepalceValues(f1, f2, spath)

    '''检查删除行是否成功！'''
    # fpath1 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\buchong\question'
    # fpath2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\More2One\buchong\replaceresults'
    # DelNanValues(fpath1, fpath2)

    '''SumTime(path)'''
    # timepath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\filtertime'
    # sp = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results'
    # SumTime(timepath, 14, sp)

    '''按列名排序'''
    # path = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\completeresults\BurakFilter'
    # spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\completeresults\SortBF'

    # path = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\completeresults\PeterFilter'
    # spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\completeresults\SortPDF'
    # SortColumns(path, spath)

    '''替换某行'''
    # spath0 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\completeresults\RepBF'
    # tpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\BurakFiltertime.csv'
    # tpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\PeterFiltertime.csv''
    # spath0 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\completeresults\RepPDF'
    # ReplaceColumns(spath, tpath, spath0)

    '''合并文件'''
    fpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\Relink_csv'
    spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\Relink_csv'
    # Concattrainfiles(fpath, 'exp_', spath)
    # fpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\aeeem_experiment\Duptotaldata'
    # spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\aeeem_experiment\ConDtdata'
    # Concattrainfiles(fpath, 'exp_', spath)

    f1 = r'D:\PycharmProjects\data_preprocessing\CK_handle\HeBurakFilter\1\13datasets'
    f2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\HeBurakFilter\2'
    # Concatfiles(f1, 'Results_HeBurakFilter.csv',  f2)
    f3 = r'D:\PycharmProjects\data_preprocessing\CK_handle\HeBurakFilter\rank'
    # SortColumns(f2, f3)
    f4 = r'D:\PycharmProjects\data_preprocessing\CK_handle\HeBurakFilter\time'
    f5 = r'D:\PycharmProjects\data_preprocessing\CK_handle\HeBurakFilter\sum_time'
    # SumTime(f4, 3, f5)
    f6 = r'D:\PycharmProjects\data_preprocessing\CK_handle\HeBurakFilter\rank_and_time'
    # ReplaceColumns(f3, f5, f6)

    path = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\CK_Mean_eachtef_allf_eachmr\select'
    spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\CK_Mean_eachtef_allf_eachmr\select_rep'
    # ReplaceColumns2(path, 'HePeterFilter', spath)

    fpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\CK_Mean_eachtef_allf_eachmr'
    f7 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\CK_Mean_eachtef_allf_eachmr'
    # RankValues(fpath+'\\' + 'AUC', 'AUC', int(7), f7)  # 7种对比方法
    # RankValues(fpath + '\\' + 'Balance', 'Balance', int(7), f7)
    # RankValues(fpath + '\\' + 'G-Mean', 'GMean', int(7), f7)
    # RankValues(fpath + '\\' + 'G-Measure', 'GMeasure', int(7),  f7)

    '''检查目标集的数量和文件中数据的条数'''
    p1 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\20190708\mean_release\rank_release _'
    p2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\20190708\mean_instance\rank_instance_'
    # Checktestfile(p1)
    # Checktestfile(p2)

    # CheckNAN('AUC',  'Unnamed: 0', 3, p2)
    # CheckNAN('AUC', 'testfile', 3, p1)
    # CheckNAN('MCC', 'testfile', 3, p2)
    # CheckNAN('MCC', 'testfile: 0', 3, p1)
    valueslist = ['ant-1.3.csv', 'camel-1.0.csv', 'e-learning-1.csv', 'prop-6.csv', 'redaktor-1.csv']
    fp1 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF'
    SP1 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF_HPF'
    # DelSomeRowValues(fp1, valueslist, SP1)

    # fp2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\rank_release_except_HPF_pdvalues\d_release_file\nb\RLiF'
    # SP2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\rank_release_except_HPF_pdvalues\d_release_file\nb'
    # Concatfiles(fp2, 'dvalues_release_RLiFFilter_NB.csv', SP2)
    # fp3 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\rank_release_except_HPF_pdvalues\d_release_file\nb\RUiF'
    # Concatfiles(fp3, 'dvalues_release_RUiFFilter_NB.csv', SP2)
    #
    # fp4 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\rank_release_except_HPF_pdvalues\p_release_file\NB\RUiF'
    # SP3 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\rank_release_except_HPF_pdvalues\p_release_file\NB'
    # Concatfiles(fp4, 'pvalues_release_RUiFFilter_NB.csv', SP3)
    # fp5 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\rank_release_except_HPF_pdvalues\p_release_file\NB\RLiF'
    # Concatfiles(fp5, 'pvalues_release_RLiFFilter_NB.csv', SP3)

    # fp6 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF_HPF_pdvalues\d_release_file\NB\RUiF'
    # SP4 = R'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF_HPF_pdvalues\d_release_file\NB'
    # Concatfiles(fp6, 'dvalues_release_RUiFFilter_NB.csv', SP4)
    # fp7 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF_HPF_pdvalues\d_release_file\NB\RLiF'
    # Concatfiles(fp7, 'pdvalues_release_RLiFFilter_NB.csv', SP4)
    #
    # fp8 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF_HPF_pdvalues\p_release_file\NB\RUiF'
    SP5 = R'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF_HPF_pdvalues\p_release_file\NB'
    # Concatfiles(fp8, 'pvalues_release_RUiFFilter_NB.csv', SP5)
    fp9 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF_HPF_pdvalues\p_release_file\NB\RLiF'
    # Concatfiles(fp9, 'pvalues_release_RLiFFilter_NB.csv', SP5)
    # St = ['DT', 'KNN', 'RF', 'MLP', 'LR', 'NB']

    '''release'''
    # basepath = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\rank_release_except_HPF_pdvalues'
    # basepath = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF_HPF_pdvalues'
    # for s in strings:
    #     folderlist = os.listdir(basepath + '\\' + s)
    #     for folder in folderlist:
    #         # filelist = os.listdir(basepath + '\\' + s + '\\' + folder)
    #         # for f in filelist:
    #         #     Concatfiles(basepath + '\\' + s + '\\' + folder + '\\' + f,  s + '_' + f, basepath)
    '''instance'''
    # basepath = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_instance'
    # # basepath = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_instance\RULiF_HPF_pdvalues'
    # strings = ['dvalues_tefile', 'pvalues_tefile']
    # for s in strings:
    #     folderlist = os.listdir(basepath + '\\' + s)  # IUiF， ILiF
    #     for folder in folderlist:
    #         filelist = os.listdir(basepath + '\\' + s + '\\' + folder)  # 分类器
    #         for f in filelist:
    #             Concatfiles(basepath + '\\' + s + '\\' + folder + '\\' + f, s + '_' + folder + '_' + f + '.csv',
    #                         basepath)

    # bp = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\rank_release_except_HPF_pdvalues'
    bp = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF_HPF_pdvalues'
    fpath1 = bp + '/' + 'd_release_classifier'
    # Concatfiles(fpath1, 'd_release_classifier.csv', bp)
    fpath2 = bp + '/' + 'p_release_classifier'
    # Concatfiles(fpath2, 'p_release_classifier.csv', bp)
    # s1 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\rank_release_except_HPF_pdvalues'
    # s2 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\RULiF_HPF_pdvalues'
    # s3 = r'D:\PycharmProjects\data_preprocessing\CK_handle\CK_results\mean_release\six_clf_20190606\pdvalues'
    # ConcatOneColumn(s1, s2, s3)

    '''参数讨论'''
    # bpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\parameters_discussions'
    # fpl = os.listdir(bpath)
    # sapath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\paremeters_mean'
    # for fp in fpl:
    #     Concatmorefiles(bpath + '\\' + fp, 0, 30, sapath)

    '''重命名文件'''
    # oldpath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\1'
    # fpl = os.listdir(oldpath)
    # spath = r'D:\PycharmProjects\data_preprocessing\CK_handle\process_results\compareresults\2'
    # for fp in fpl:
    #     RenameFiles(oldpath + '\\' + fp, spath)

    endtime = time.clock()  # 计时结束
    totaltime = endtime - starttime

    print('程序结束运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序共运行时间为:', totaltime)
