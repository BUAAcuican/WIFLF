#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
Answering RQ3 for Paper:
Weighted Isolation Forest Filter (Combined with SMOTE)
for Instance-based Cross-Project Defect Prediction
============================================================================
@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
@2020/10/20
============================================================================
"""
# print(__doc__)
from __future__ import print_function

# print(__doc__)
import scipy
import pandas as pd
import numpy as np
import math
import heapq
import random
import copy
import time
import os
import multiprocessing
from functools import partial


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
    for file in file_list:
        data = pd.read_csv(fpath + '\\' + file, encoding='gbk' )
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
        df_std.index = namelist + ['std' ]
        df_median.index = namelist + ['median']


        df_mean.to_csv(savepath + '\\' + file.rstrip('.csv') + '_mean.csv')
        df_std.to_csv(savepath + '\\' + file.rstrip('.csv') + '_std.csv')
        df_median.to_csv(savepath + '\\' + file.rstrip('.csv') + '_median.csv')
        # print(df_mean.tail(3), len(df_mean))

def Multi_Aver_Measures(fpath, spath, string0, measure_index):
    """
    Multi_Aver_Measures() details go here：
    Contact multi-approaches for one measure of every and the whole datasets
    :param fpath: type: string, absolute path of multi-folders, eg. xxx is the fpath, .csv is like below:
                  xxxx/path1/path2/[xxx.csv,...,xxx.csv]
    :param spath: type: string, absolute path of multi-approaches results of one measure
    :param string0: type: string, the classifier and the value name, such as 'LR_mean', 'NB_std', 'RF_median'
    :param measure_index: type: string, the name of measures, such as, 'Balance', 'MCC',
                       measures are in [AUC	Balance	MCC	G-Measure ...TP	TN	FP	FN  train_len_before
                                train_len_after	test_len runtime clf testfile trainfile runtimes]
    :return: Null
    """

    global index_name
    folder1list = os.listdir(fpath)  # the first layer folder, data2020内所有的文件夹
    for folder1 in folder1list:
        if "_mean" in folder1:
            folder2list = os.listdir(fpath + '\\' + folder1)  # the second layer folder， 所有“_mean”的文件夹
            M2Olist = []
            O2Olist = []
            columns_name = []
            for folder2 in folder2list:
                if folder2.split('_')[0] == 'M2O':
                    M2Olist.append(folder2)
                    columns_name.append(folder2.rstrip('.csv'))
                elif folder2.split('_')[0] == 'O2O':
                    O2Olist.append(folder2)
                    columns_name.append(folder2.rstrip('.csv'))

            # 将“_mean"中前缀M20/O2O的文件夹中后缀_LR_mean/_LR_std/_LR_median/……相同的合并
            senarios = [M2Olist,  O2Olist]
            # print(senarios)
            senarios_name = ['M2O', 'O2O']
            for k in range(len(senarios)):
                if senarios[k] != []:
                    filelist0 = os.listdir(fpath + '\\' + folder1 + '\\' + senarios[k][0])
                    pd_measure = pd.DataFrame()
                    columns_name = []
                    for i in range(len(filelist0)):  # all files in filelist0
                        if string0 in filelist0[i]:
                            print('the index of string0 in folder0:', i, folder1, filelist0[i])  # verify the folder is same as the following or not
                            columns_name.append(filelist0[i].rstrip('.csv'))  # the name of approach
                            readpath0 = fpath + '\\' + folder1 + '\\' + senarios[k][0] + '\\' + filelist0[i]
                            df1 = pd.read_csv(readpath0)
                            pd_measure = pd.concat([pd_measure, pd.DataFrame(df1[measure_index])],
                                                   axis=1, sort=False, ignore_index=False)
                            index_name = df1.iloc[:, 0]
                            for j in range(1, len(senarios[k]), 1):

                                filelistj = os.listdir(fpath + '\\' + folder1 + '\\' + senarios[k][j])
                                # print(filelistj)
                                for m in range(len(filelistj)):  # all files in filelist0
                                    if string0 in filelistj[m]:
                                        readpath1 = fpath + '\\' + folder1 + '\\' + senarios[k][j] + '\\' + filelistj[m]
                                        columns_name.append(filelistj[m].rstrip('.csv'))  #
                                        print('the index of string0 in folderj:', j, filelistj[m])  # filelistj[m]
                                        df2 = pd.read_csv(readpath1)
                                        pd_measure = pd.concat([pd_measure, pd.DataFrame(df2[measure_index])],
                                                               axis=1, sort=False, ignore_index=False)


                    name = folder1.rstrip('mean') + senarios_name[k] + '_' + string0 + '_' + measure_index + '.csv'
                    # print(columns_name)
                    savepath = spath + '\\MSM'
                    if not os.path.exists(savepath):
                        os.mkdir(savepath)
                    pd_measure.columns = columns_name
                    # print(pd_measure.tail(4), pd_measure.index, index_name)
                    pd_measure.index = index_name

                    pd_measure.to_csv(savepath + '\\' + name)


def Average_Mean(rootfolders, spaths, stringlist, measurelist):
    """
    Func Average_Mean details go here:
    Call Func Average_Measures() and Func Multi_Aver_Measures()
    :param rootfolders: type, string list, absolute path list of different dataset results, such as xxxx\AEEEM, XXXX\Relink
    :param spaths: type, string list,
    :param stringlist:
    :param measurelist:
    :return:
    """

    for i in range(len(rootfolders)):
        fl = os.listdir(rootfolders[i])
        string = 'testfile'
        for folder in fl:
            if folder not in ['data', 'earlier', 'newer']:
                fpath = rootfolders[i] + "\\" + folder
                if not os.path.exists(spaths[i]):
                    os.mkdir(spaths[i])
                Average_Measures(fpath, spaths[i], string)

    rootfolder = rootfolders[0].rstrip(r'\AEEEM')
    for string in stringlist:
        for measure in measurelist:
            Multi_Aver_Measures(rootfolder, rootfolder, string, measure)


def ContactFiles(fpath, spath, strings, mode, clf, measure_name, convert=True):
    """
    Func ContactFiles details go here:
    1) Read files from fpath, The measure file name is named as 'dataset_mode_method_clf_string_measure.csv',
       e.g., AEEEM_M20_LR_mean_MCC.csv;
    2) According to the conditions of "strings, mode, clf, measure_name" to combined the values.
       E.g., the mean and std values of one measure from the same dataset and clf are combined.
       The columns names are like HSBF_mean, HSBF_STD, BF_mean, BF_std....
    3) Save the combined results in spath.
    :param fpath: type: string, absolute path of measure files, such as D:\RQ2\LR
    :param spath: type: string,absolute path of save concacted files, such as D:\RQ2\handle
    :param strings: type: string list, concact two files names strings, ['mean', 'std']
    :param mode: type: string, the senario for conduct models, i.e., 'M2O' or 'O2O'
    :param clf: type: string, 'LR' or 'NB' OR 'RF'
    :param measure_name: type: string, measure name, such as 'Balance', 'MCC'.
    :param convert: default = True. True means the 'mean($\pm$std)'
    :return:
    """

    filelist = os.listdir(fpath)

    # e.g. CK_M2O_LR_xx_AUC
    measure_filelist = []
    for file in filelist:
        # measure_name:AUC/Balance/G-Mean2/G-Measure/MCC
        # clf: LR/NB/RF
        # mode: M2O/O2O

        # s = file.rstrip('.csv').split('_')[-1]
        if file.rstrip('.csv').split('_')[-1] == measure_name \
                and file.split('_')[2] == clf \
                and file.split('_')[1] == mode:
            measure_filelist.append(file)
    if measure_filelist != []:
        print('measure filelist name:', measure_filelist)

    for file1 in measure_filelist:
        for file2 in measure_filelist:
            ########## step1: concact "std" and "mean"  ##########
            if strings[1] in file2 and strings[0] in file1 and file1.split('_')[0] == file2.split('_')[0]:
                df_mean = pd.read_csv(fpath + "\\" + file1)
                index = df_mean.iloc[:, 0].values  # df_mean has default index
                # print('index name:', index)
                df_std = pd.read_csv(fpath + '\\' + file2)
                print('concact file names:', file1, file2)
                k = 1
                for i in range(1, df_mean.shape[1], 1):
                    # col_name = df_mean.columns.tolist()
                    # df_mean.insert
                    df_mean.insert(i+k, df_std.columns.values[i], df_std.iloc[:, i], allow_duplicates=True)
                    k = k+1
                # df_mean.iloc[]
                # print(file1.split('_')[0:2], file1.split('_')[-1])
                # print(df_mean.index)
                df_mean = df_mean * 100
                dfms4 = df_mean.iloc[:, 1:].applymap("{0:.02f}".format)  # or lambda x:'%.2f' % x
                dfms4.index = index

                # print(dfms4)
                # dfms4.to_csv(spath + '/' + file1.split('_')[0] + '_' +
                #              file1.split('_')[1] + '_' + file1.split('_')[-1], index=True)
                if convert:
                    ########## step2: mean(±std)  ##########
                    column_name = dfms4.columns.values
                    dfms4.astype(str)
                    # dfms4['signal2'] = pd.DataFrame(np.nan * len(index))
                    df_msm = pd.DataFrame()
                    for j in range(0, len(column_name) - 1, 2):
                        # for RQ2
                        if column_name[0].split('_')[0] != 'CPDP':
                            temp_name = column_name[j + 1].split('_')[0]
                            df_msm[temp_name] = dfms4[[column_name[j], column_name[j + 1]]].apply(
                                lambda x: '($\pm$'.join(x), axis=1)
                        else:  # for RQ1
                            temp_name = column_name[j + 1].rstrip(column_name[j + 1].split('_')[-1])
                            df_msm[temp_name] = dfms4[[column_name[j], column_name[j + 1]]].apply(
                                lambda x: '($\pm$'.join(x), axis=1)

                        # df_msm[temp_name] = df_msm[[temp_name, 'signal2']].apply(lambda x: ')'.join(x), axis=1)
                    # print(df_msm)
                    df_msm = df_msm + ')&'  # add ')' in each column #

                    df_msm.to_csv(spath + '/' + file1.split('_')[0] + '_' +
                                  file1.split('_')[1] + '_' +
                                  file1.split('_')[2] + '_' + file1.split('_')[-1], index=True)
                else:
                    df_msm = dfms4
                    df_msm.to_csv(spath + '/' + file1.split('_')[0] + '_' +
                                  file1.split('_')[1] + '_' +
                                  file1.split('_')[2] + '_' + file1.split('_')[-1], index=True)


def ContactMSM(fpath, spath, strings, modes, clfs, measure_names):
    """
    Func ContactMSM details go here:
    1) Call Func ContactFiles and make the combined different measures by different conditions
    2) Save the multiple files of different combined results as xx\\xx\\handle\\xx.csv.
    :param fpath: type: string, absolute path of different clfs folders which contain measure files, such as D:\RQ2
    :param spath: type: string, absolute path of a save folder which contains concacted files, such as D:\RQ2
    :param strings: type: string list, concact two files names strings, ['mean', 'std']
    :param modes:  type: string list, the senario for conduct models, i.e., ['M2O' , 'O2O']
    :param clfs: type: string list, ['LR', 'NB', 'RF']
    :param measure_names: type: string list, measure names, such as ['Balance', 'MCC', 'G-Measure', 'G-Mean2'].
    :return:
    """

    global savepath

    folderlist = os.listdir(fpath)
    for folder in folderlist:

        if folder == 'handle':
            savepath = spath + '\\' + folder

        elif folder != 'handle':
            readpath = fpath + '\\' + folder
            for mode in modes:
                for clf in clfs:
                    for measure_name in measure_names:
                        ContactFiles(readpath, savepath, strings, mode, clf, measure_name)

def Concat_More_Files(fpath, spath, stringlist=['Balance', 'MCC','G-Measure','G-Mean2']):
    '''

    :param fpath:
    :param spath:
    :return:
    '''
    filelist = os.listdir(fpath)
    if not os.path.exists(spath):
        os.mkdir(spath)
    lrlist = []
    rflist = []
    nblist = []
    for file in filelist:
        if file.split('_')[-1] == 'LR.csv':
            lrlist.append(file)
        elif file.split('_')[-1] == 'RF.csv':
            rflist.append(file)
        elif file.split('_')[-1] == 'NB.csv':
            nblist.append(file)
        else:
            print('The names of files are without LR/NB/RF.')

    for filelist_each in [lrlist, nblist, rflist]:

        df_all = pd.DataFrame()
        for i in range(len(filelist_each)):
            # df0 = pd.read_csv(fpath + '\\' + filelist_each[0])
            df = pd.read_csv(fpath + '\\' + filelist_each[i])
            if fpath.split('\\')[-1] == 'O2O' and len(df) == int(150):
                df = pd.concat([df, df, df, df], axis=0)
            elif fpath.split('\\')[-1] == 'O2O' and len(df) == int(420):
                # df = pd.read_csv(fpath + '\\' + filelist_each[i])
                df = pd.concat([df, df, df, df, df,
                                df, df, df, df, df,
                                df, df, df], axis=0)
            elif fpath.split('\\')[-1] == 'O2O' and len(df) == int(90):
                df = pd.concat([df, df], axis=0)
            else:
                pass
            file_each = filelist_each[i]
            df_sort = df.sort_values(by=['testfile', 'runtimes'], ascending=True)
            df_sort.drop(columns=[df_sort.columns.values[0]], inplace=True)
            df_sort.set_index(["testfile"], inplace=True)
            df_sort.to_csv(spath + '\\' + file_each)
            df_measures = df_sort[stringlist]

            print("name of the concact file:", file_each)
            print(df_measures.shape)
            df_all = pd.concat([df_all, df_measures], axis=1)
            print(df_all.size, df_measures.size)
            # df_all = merge(df_all, df_measures, left_index=True, right_index=True, how='outer')

        df_all.to_csv(spath + '\\' + 'all_' + filelist_each[0].split('_')[-1])


def ConcatFile2(fp, sp):
    fp = r'D:\PycharmProjects\cuican\data2020\MSM\RQ2\handle\M2O'
    fl = os.listdir(fp)
    filelist = []
    for f in fl:
        if ".csv" in f:
            filelist.append(f)

    for i in range(9):
        df = pd.DataFrame()
        for file in filelist:
            if file.split('_')[2] == filelist[i].split('_')[2] \
                    and file.split('_')[3] == filelist[i].split('_')[3]:
                df1 = pd.read_csv(fp + '\\' + file)
                df = pd.concat([df, df1], axis=0, ignore_index=True)
        filename = filelist[i].split('_')[1] + "_" + filelist[i].split('_')[2] + "_" + filelist[i].split('_')[3]

        df.to_csv(r'D:\PycharmProjects\cuican\data2020\MSM\RQ2\handle' + '\\'
                  + filename)

if __name__ == '__main__':

    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # starttime = time.clock()  # 计时开始
    starttime = time.perf_counter()
    # test Average_Measures
    # fpath_test = r'D:\PycharmProjects\cuican\data2020\test\O2O_CPDP_CPDP_iForest_alpha0.6'
    # spath_test = r'D:\PycharmProjects\cuican\data2020\test'
    # string = 'testfile'
    # Average_Measures(fpath_test, spath_test, string)

    # RQ1
    # rootfolder = r'D:\PycharmProjects\cuican\data2020\RQ1'
    # spath = r'D:\PycharmProjects\cuican\data2020\RQ1_mean'
    # fl = os.listdir(rootfolder)
    # string = 'testfile'
    # for folder in fl:
    #     if folder not in ['data', 'earlier', 'newer', 'o2o']:
    #         fpath = rootfolder + '\\' + folder
    #         Average_Measures(fpath, spath, string)

    # # # RQ2
    # rootfolders = [r'D:\PycharmProjects\cuican\data2020\AEEEM',
    #                r'D:\PycharmProjects\cuican\data2020\Relink',
    #                r'D:\PycharmProjects\cuican\data2020\CK',
    #                r'D:\PycharmProjects\cuican\data2020\CK2']
    # spaths = [r'D:\PycharmProjects\cuican\data2020\AEEEM_mean',
    #           r'D:\PycharmProjects\cuican\data2020\Relink_mean',
    #           r'D:\PycharmProjects\cuican\data2020\CK_mean',
    #           r'D:\PycharmProjects\cuican\data2020\CK2_mean']
    # for i in range(len(rootfolders)):
    #     fl = os.listdir(rootfolders[i])
    #     string = 'testfile'
    #     for folder in fl:
    #         if folder not in ['data', 'earlier', 'newer']:
    #             fpath = rootfolders[i] + '\\' + folder
    #             if not os.path.exists(spaths[i]):
    #                 os.mkdir(spaths[i])
    #             Average_Measures(fpath, spaths[i], string)

    # ############# test Func Multi_Aver_Measures  ##############
    # rootfolder = r'D:\PycharmProjects\cuican\data2020'
    # spath = r'D:\PycharmProjects\cuican\data2020'
    # stringlist = ['LR_mean', 'NB_mean', 'RF_mean',
    #               'LR_std', 'NB_std', 'RF_std',
    #               'LR_median', 'NB_median', 'RF_median']
    # measurelist = ['Balance', 'MCC', 'G-Measure',
    #                'G-Mean2', 'AUC', 'PD(Recall, TPR)', 'PF(FPR)']
    # Multi_Aver_Measures(rootfolder, spath, stringlist[0], measurelist[0])

    ############# test Func Average_Mean  ##############
    # rootfolders = [r'D:\PycharmProjects\cuican\data2020\AEEEM',
    #                r'D:\PycharmProjects\cuican\data2020\Relink',
    #                r'D:\PycharmProjects\cuican\data2020\CK',
    #                r'D:\PycharmProjects\cuican\data2020\CK2']
    # spaths = [r'D:\PycharmProjects\cuican\data2020\AEEEM_mean',
    #           r'D:\PycharmProjects\cuican\data2020\Relink_mean',
    #           r'D:\PycharmProjects\cuican\data2020\CK_mean',
    #           r'D:\PycharmProjects\cuican\data2020\CK2_mean']
    # stringlist = ['LR_mean', 'NB_mean', 'RF_mean',
    #               'LR_std', 'NB_std', 'RF_std',
    #               'LR_median', 'NB_median', 'RF_median']
    # measurelist = ['Balance', 'MCC', 'G-Measure',
    #                'G-Mean2', 'AUC', 'PD(Recall, TPR)', 'PF(FPR)']
    # Average_Mean(rootfolders, spaths, stringlist, measurelist)

    ############# test Func ContactFiles  ##############
    # fpath = r'D:\PycharmProjects\cuican\data2020\MSM\RQ2\LR'
    # spath = r'D:\PycharmProjects\cuican\data2020\MSM\RQ2\handle'
    # measure_name:AUC/Balance/G-Mean2/G-Measure/MCC
    # clf: LR/NB/RF
    # mode: M2O/O2O
    # fpath = r'D:\PycharmProjects\cuican\data2020\MSM\RQ1_M2O_mean'
    # spath = r'D:\PycharmProjects\cuican\data2020\MSM\RQ1_M2O_handle'
    # strings = ['mean', 'std']
    # mode = 'M2O'
    # clf = 'LR'
    # measure_name = 'Balance'
    # ContactFiles(fpath,  spath, strings, mode, clf, measure_name)

    # ### to gain mean($\pm$std)
    # # for RQ2
    # modes = ['M2O', 'O2O']
    # fpath = r'D:\PycharmProjects\cuican\data2020\MSM\RQ2'
    # spath = r'D:\PycharmProjects\cuican\data2020\MSM\RQ2'
    # # # for RQ1
    # fpath = r'D:\PycharmProjects\cuican\data2020\MSM\RQ1'
    # spath = r'D:\PycharmProjects\cuican\data2020\MSM\RQ1'
    # modes = ['M2O']
    # clfs = ['LR', 'RF']
    # measure_names = ['Balance', 'G-Mean2', 'G-Measure']
    # # modes = ['O2O']
    # # clfs = ['LR', 'NB', 'RF']
    # # measure_names = ['Balance', 'MCC', 'G-Mean2', 'G-Measure']
    # strings = ['mean', 'std']
    # ContactMSM(fpath, spath, strings, modes, clfs, measure_names)

    ########## test Concat_More_Files() #############
    # fpath = r'D:\PycharmProjects\cuican\data2020\ACR\Relink\M2O'
    # spath = r'D:\PycharmProjects\cuican\data2020\ACR\Relink\concact_M2O'
    # fpath = r'D:\PycharmProjects\cuican\data2020\ACR\RQ1\O2O'
    # spath = r'D:\PycharmProjects\cuican\data2020\ACR\RQ1\concact_O2O'
    # Concat_More_Files(fpath, spath, stringlist=['Balance', 'MCC', 'G-Measure', 'G-Mean2'])

    # concact many measures of more approaches for each dataset group into a file
    # fps = r'D:\PycharmProjects\cuican\data2020\ACR'
    # fpaths = os.listdir(fps)  # aeeem, relink ,ck, ck2
    # for fpath in fpaths:  # aeeem
    #     print('当前路径：', fpath)
    #     folders = os.listdir(fps + '\\' + fpath)  # m2o,o2o,concat_m2o,concat_o2o
    #     for folder in folders:
    #         if '_' not in folder:
    #             fp = fps + '\\' + fpath + '\\' + folder
    #             sp = fps + '\\' + fpath + '\\concact_' + fp.split('\\')[-1]
    #             Concat_More_Files(fp, sp, stringlist=['Balance', 'MCC', 'G-Measure', 'G-Mean2'])





    # endtime = time.clock()  # 计时结束
    endtime = time.perf_counter()
    totaltime = endtime - starttime
    print('程序结束运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序共运行时间为:', totaltime)


