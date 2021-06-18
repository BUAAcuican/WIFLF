
#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import sys
import time
import copy
import heapq
from collections import Counter
import os
import scipy.stats as stats
from statistics import mean, stdev  # version >= Python3.4

def stepMethodMean():
    pwd = os.getcwd()
    # print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    # datapath = father_path + '/dataset-inOne/'
    # resultspath = father_path + '/results/best_parameters/'
    # spath = father_path + '/handleresults/best_parameters/'
    resultspath = father_path + '/paperResults/Disscusion1/'
    spath = father_path + '/paperResults/Dis1_Mean/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    csvfiles = os.listdir(resultspath)
    csvfilenum = len(csvfiles)
    meanfile_lr = pd.DataFrame()
    meanfile_rf = pd.DataFrame()
    meanfile_nb = pd.DataFrame()
    rownames_lr = []
    rownames_nb = []
    rownames_rf = []
    for i in range(len(csvfiles)):
        csvfile = csvfiles[i]
        if "LR" in csvfile:
            df_lr = pd.read_csv(resultspath + csvfile)
            values_lr = df_lr.iloc[:, 0:23]
            mean_lr = pd.DataFrame(values_lr.mean())
            meanfile_lr = pd.concat([meanfile_lr, mean_lr.T])
            rownames_lr.append(csvfile.rstrip('.csv'))
        if "RF" in csvfile:
            df_rf = pd.read_csv(resultspath + csvfile)
            values_rf = df_rf.iloc[:, 0:23]
            mean_rf = pd.DataFrame(values_rf.mean())
            meanfile_rf = pd.concat([meanfile_rf, mean_rf.T])
            rownames_rf.append(csvfile.rstrip('.csv'))
        if "NB" in csvfile:
            df_nb = pd.read_csv(resultspath + csvfile)
            values_nb = df_nb.iloc[:, 0:23]
            mean_nb = pd.DataFrame(values_nb.mean())
            meanfile_nb = pd.concat([meanfile_nb, mean_nb.T])
            rownames_nb.append(csvfile.rstrip('.csv'))

    meanfile_lr.index = rownames_lr
    meanfile_lr.to_csv(spath + 'stepMethodMean_LR.csv')

    meanfile_rf.index = rownames_rf
    meanfile_rf.to_csv(spath + 'stepMethodMean_RF.csv')

    meanfile_nb.index = rownames_nb
    meanfile_nb.to_csv(spath + 'stepMethodMean_NB.csv')
    # print(meanfile_nb, mean_nb)

def concactDifferentDatasetsFiles(methodnum=7, learnernum=3):
    """
    #####temp function####
    FUNCITON CONCACTDIFFERENTDATASETSFILES() SUMMARY GOES HERE:
    Concact all the results from different datasets and different leaners to receive a csvfile.
    The obtained csvfile is sorted according to the columns "testfile" and "runtimes".
     e.g., "../AEEEM/M2O_CPDP_Bellwether/Bellwether_Naive_LR.csv" &
    “../PROMOSE_earlier/M2O_CPDP_Bellwether/Bellwether_Naive_LR.csv” to concact the same name csv files, and
    get a new "../concactFiles/Bellwether_Naive_LR.csv".
    Note: The number of *l* and *m* in the code are changing if the number of methods and learners change.
    @param methodnum: type: int, default=7, the number of methods
    @param learnernum: type: int, default=3, the number of classifiers
    @return None
    """
    pwd = os.getcwd()
    # print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    resultspath = father_path + '/M2O_allResults/'
    spath = father_path + '/concactFiles/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    directoryfiles = os.listdir(resultspath)
    directoryfilenum = len(directoryfiles)

    # l: the index of the method, type: int, range: from 0 to len(subdirectoryfiles1)-1
    for l in range(methodnum):  # number of method
        # m  the index of the learner, type: int, range: from 0 to len(csvfiles1)-1
        for m in range(learnernum):  # number of learner
            i = 0
            directoryfile1 = directoryfiles[i]  # e.g, AEEEM
            subdirectoryfiles1 = os.listdir(resultspath + str(directoryfile1))   # ['M2O_CPDP_Bellwether', 'xxxTNB']
            csvfiles1 = os.listdir(resultspath + directoryfile1 + "/" + subdirectoryfiles1[l])  # ["Bellwether_Naive_LR.csv", "Bellwether_Naive_NB.csv", "Bellwether_Naive_RF.csv", ]
            csvfile1 = resultspath + directoryfile1 + "/" + subdirectoryfiles1[l] + "/" + csvfiles1[m]
            dfm = pd.read_csv(csvfile1)  # “../AEEEM/M2O_CPDP_Bellwether/Bellwether_Naive_LR.csv”
            # dfm.sort(columns=['testfile'], axis=0, ascending=True)
            dfm = dfm.sort_values(by=["testfile", "runtimes"], axis=0, ascending=True)
            for j in range(directoryfilenum):
                if j != i:
                    directoryfile2 = directoryfiles[j]  # e.g, PROMISE_earlier
                    subdirectoryfiles2 = os.listdir(resultspath + directoryfile2)   # ['M2O_CPDP_Bellwether', 'xxxTNB']
                    for p in range(len(subdirectoryfiles2)):
                        if subdirectoryfiles1[l] == subdirectoryfiles2[p]:
                            csvfiles2 = os.listdir(resultspath + directoryfile2 + "/" + subdirectoryfiles2[p])  # ["Bellwether_Naive_LR.csv", "Bellwether_Naive_NB.csv", "Bellwether_Naive_RF.csv"]
                            for n in range(len(csvfiles2)):
                                if csvfiles1[m] == csvfiles2[n]:
                                    csvfile2 = resultspath + directoryfile2 + "/" + subdirectoryfiles2[p] + "/" + csvfiles2[n]  # “../PROMOSE_earlier/M2O_CPDP_Bellwether/Bellwether_Naive_LR.csv”
                                    dfn = pd.read_csv(csvfile2)
                                    # dfn.sort(columns=['testfile'], axis=0, ascending=True)
                                    dfn = dfn.sort_values(by=["testfile", "runtimes"], axis=0, ascending=True)
                                    dfm = pd.concat([dfm, dfn])
            dfm.to_csv(spath + "/" + csvfiles1[m], index=False)

def MeanMedianStd(measureindex=1, learnernum=3):
    """
    Function MEANMEDIANSTD() SUMMARY GOES HERE:
    Calculate the mean, median and standard
    @param learnernum: number of classifiers, type: int, default=3, means:NB, LR, RF
    @param measureindex: index of measure in the measure csv file, type: int, default=1, means: AUC
    @return: mean, median and standard deviation of each dataset and the mean and median values of all datasets.
    """
    pwd = os.getcwd()
    # print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    # resultspath = father_path + '/Results/'   # the first column is "Unnamed: 0"
    # spath = father_path + '/MeanStdMedianFiles/'
    # spath1 = father_path + '/MethodsMeanStdMedianFiles/'

    resultspath = father_path + '/paperResults/Disscusion1/'  # the first column is "Unnamed: 0"
    spath = father_path + '/paperResults/Dis1_MeanStdMedianFiles/'
    spath1 = father_path + '/paperResults/Dis1_MethodsMeanStdMedianFiles/'

    if not os.path.exists(spath):
        os.mkdir(spath)
    if not os.path.exists(spath1):
        os.mkdir(spath1)

    csvfiles = os.listdir(resultspath)

    for i in range(learnernum):
        columnname = [csvfiles[i].rstrip(".csv")]
        dfi = pd.read_csv(resultspath + csvfiles[i])
        dfi = dfi.replace(np.nan, 0)  # replace the NAN to 0
        learnername = csvfiles[i].split("_")[-1].rstrip(".csv")

        index = []
        for ind in range(0, len(dfi), 30):
            tempindex = dfi["testfile"][ind]
            if ".csv" in tempindex:
                tempindex = tempindex.rstrip(".csv")
            index.append(tempindex)

        dfiMean = pd.DataFrame()
        dfiMedian = pd.DataFrame()
        dfiStd = pd.DataFrame()

        for m in range(0, len(dfi), 30):
            tempdfiMean = pd.DataFrame(dfi.iloc[m:m+30, 1:].mean())  # the first column is "Unnamed: 0"
            dfiMean = pd.concat([dfiMean, tempdfiMean.T])
            
            tempdfiStd = pd.DataFrame(dfi.iloc[m:m+30, 1:].std())
            dfiStd = pd.concat([dfiStd, tempdfiStd.T])

            tempdfiMedian = pd.DataFrame(dfi.iloc[m:m + 30, 1:].median())
            dfiMedian = pd.concat([dfiMedian, tempdfiMedian.T])

        dfiMean.index = index
        dfiStd.index = index
        dfiMedian.index = index

        # spath = father_path + '/MeanStdMedianFiles/'
        dfiMean.to_csv(spath + "Mean_" + csvfiles[i], index=True)
        dfiStd.to_csv(spath + "Std_" + csvfiles[i], index=True)
        dfiMedian.to_csv(spath + "Median_" + csvfiles[i], index=True)

        dfMeasureMean = pd.DataFrame(dfiMean.iloc[:, measureindex])
        dfMeasureStd = pd.DataFrame(dfiStd.iloc[:, measureindex])
        dfMeasureMedian = pd.DataFrame(dfiMedian.iloc[:, measureindex])


        formatMean = (lambda x: '%.2f' % (x * 100))  # value * 100 and keep 2 demicals
        dfMeasureMeanStd = dfMeasureMean.applymap(formatMean)
        formatStd = (lambda x: ('%.2f') % float(x))  # .2%
        dfMeasureMeanStd = pd.concat([dfMeasureMeanStd, dfMeasureStd.applymap(formatStd)], axis=1)

        measurename = dfiMean.columns.values[measureindex]  # note: dfi.columns != dfiMean.columns

        for j in range(learnernum, len(csvfiles)):
            if j != i and csvfiles[i].split("_")[-1] == csvfiles[j].split("_")[-1]:
                dfjMean = pd.DataFrame()
                dfjStd = pd.DataFrame()
                dfjMedian = pd.DataFrame()

                columnname.append(csvfiles[j].rstrip(".csv"))
                dfj = pd.read_csv(resultspath + csvfiles[j])
                dfj = dfj.replace(np.nan, 0)

                for n in range(0, len(dfj), 30):
                    tempdfjMean = pd.DataFrame(dfj.iloc[n:n + 30, 1:].mean())  # the first column is "Unnamed: 0"
                    dfjMean = pd.concat([dfjMean, tempdfjMean.T])

                    tempdfjStd = pd.DataFrame(dfj.iloc[n:n + 30, 1:].std())
                    dfjStd = pd.concat([dfjStd, tempdfjStd.T])

                    tempdfjMedian = pd.DataFrame(dfj.iloc[n:n + 30, 1:].median())
                    dfjMedian = pd.concat([dfjMedian, tempdfjMedian.T])

                dfjMean.index = index
                dfjStd.index = index
                dfjMedian.index = index

                # spath = father_path + '/MeanStdMedianFiles/'
                dfjMean.to_csv(spath + "Mean_" + csvfiles[j], index=True)
                dfjStd.to_csv(spath + "Std_" + csvfiles[j], index=True)
                dfjMedian.to_csv(spath + "Median_" + csvfiles[j], index=True)

                tempdfMeasureMean = pd.DataFrame(dfjMean.iloc[:, measureindex])
                dfMeasureMean = pd.concat([dfMeasureMean, tempdfMeasureMean], axis=1)

                tempdfMeasureStd = pd.DataFrame(dfjStd.iloc[:, measureindex])
                dfMeasureStd = pd.concat([dfMeasureStd, tempdfMeasureStd], axis=1)

                tempdfMeasureMedian = pd.DataFrame(dfjMedian.iloc[:, measureindex])
                dfMeasureMedian = pd.concat([dfMeasureMedian, tempdfMeasureMedian], axis=1)

                dfMeasureMeanStd = pd.concat([dfMeasureMeanStd, tempdfMeasureMean.applymap(formatMean)], axis=1)
                dfMeasureMeanStd = pd.concat([dfMeasureMeanStd, tempdfMeasureStd.applymap(formatStd)], axis=1)

        dfMeasureMean.columns = columnname
        dfMeasureMedian.columns = columnname
        dfMeasureStd.columns = columnname
        columnname_2 = []
        for cn in columnname:
            columnname_2.append(cn + "_mean")
            columnname_2.append(cn + "_std")
        dfMeasureMeanStd.columns = columnname_2
        dfMeasureMeanStd.index = index

        # Adding "Average" row on all datasets
        Mean = dfMeasureMean.mean()
        dfMeasureMean = dfMeasureMean.append(Mean, ignore_index=True)  # add a row, ignore_index=True, or give a name for series index
        index.append("Average")
        dfMeasureMean.index = index

        # Adding "Median" row on all datasets
        Median = dfMeasureMedian.median()
        dfMeasureMedian = dfMeasureMedian.append(Median, ignore_index=True)
        index.remove("Average")
        index.append("Median")
        dfMeasureMedian.index = index

        #  spath1 = father_path + '/MethodsMeanStdMedianFiles/'
        dfMeasureMean.to_csv(spath1 + learnername + "_Mean_" + str(measurename) + ".csv", index=True)
        dfMeasureStd.to_csv(spath1 + learnername + "_Std_" + str(measurename) + ".csv", index=True)
        dfMeasureMedian.to_csv(spath1 + learnername + "_Median_" + str(measurename) + ".csv", index=True)
        dfMeasureMeanStd.to_csv(spath1 + learnername + "_MeanStd_" + str(measurename) + ".csv", index=True)

def WinTieLoss(data1, data2, alpha, r):
    '''
    Function WITTIELOSS() SUMMARY GOES HERE:
    Calculate win/tie/loss values at the significant level about *data1* compared with *data2*
    @param data1: the data of observation
    @param data2: the data of control
    @param alpha: the cut-off value to judge the significance, type: float, default: 0.05
    @param r: the number of the data1=data2
    @return: wtl
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

def Cohens_d(data1, data2):
    '''
    Calculate Cohen's d values about *data1* compared with *data2*
    https://blog.csdn.net/illfm/article/details/19917181
    https://cloud.tencent.com/developer/article/1634971
    @param data1: data for the observation group
    @param data2: data from the control group
    @return:d: Cohens'd value and esl: effect size level
    '''
    m1 = mean(data1)
    m2 = mean(data2)
    s1 = stdev(data1)
    s2 = stdev(data2)
    d = (m1 - m2) / (np.sqrt((s1 ** 2 + s2 ** 2) / 2))
    esl = 'None'
    if np.abs(d) >= 0.8:
        esl = 'L' #'Large(L)'
    elif np.abs(d) < 0.8 and np.abs(d) >= 0.5:
        esl = 'M'  # 'Medium(M)'
    elif np.abs(d) < 0.5 and np.abs(d) >= 0.2:
        esl = 'S'  # 'Small(S)'
    elif np.abs(d) < 0.2:
        esl = 'N'  # 'Negligible(N)'
    else:
        print('d is wrong, please check...')

    return d, esl

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

def concactMeasureofDifferentMethodsFiles(methodnum=10, learnernum=3, columnindex=1):
    """
    learnerIndex from 1, it is wrong, why? waiting for checking.
    @param methodnum: number of methods, type: int, default=11, means:
    Bellwether_Naive/BF/DFAC/HISNN/HSBF/TCA+/TNB/
    Zscore_WiFFplusL_CPDP1/Zscore_WiFFplusL_CPDP2/Zscore_WiFFplusL_CPDP3/Zscore_WiFF_CPDP

    @param learnernum: number of classifiers, type: int, default=3, means:NB, LR, RF
    @param columnindex: index of measure in the measure csv file, type: int, default=1, means: AUC
    @return:
    """
    pwd = os.getcwd()
    # print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    # resultspath = father_path + '/results/'   # the first column is "Unnamed: 0"
    # spath = father_path + '/concactMeasureofDifferentMethodsFiles/'

    resultspath = father_path + '/paperResults/Disscusion1/'  # the first column is "Unnamed: 0"
    spath = father_path + '/paperResults/Dis1_concactMeasureofDifferentMethodsFiles/'


    if not os.path.exists(spath):
        os.mkdir(spath)

    csvfiles = os.listdir(resultspath)
    # csvfilenum = len(csvfiles)

    for learnerIndex in range(learnernum):
        csvfile1 = csvfiles[learnerIndex]
        columname = []
        columname.append(csvfile1.rstrip(".csv"))
        df = pd.read_csv(resultspath + csvfile1)
        dfMeasure = pd.DataFrame(df.iloc[:, columnindex])
        for m in range(learnerIndex + learnernum, methodnum*learnernum, learnernum):
            columname.append(csvfiles[m].rstrip(".csv"))
            df2 = pd.read_csv(resultspath + csvfiles[m])
            tempdfMeasure = pd.DataFrame(df2.iloc[:, columnindex])
            dfMeasure = pd.concat([dfMeasure, tempdfMeasure], axis=1)
        meausureName = df.columns.values[columnindex]
        dfMeasure.index = df["testfile"]
        dfMeasure.columns = columname
        dfMeasure = dfMeasure.replace(np.nan, 0)  # NAN-->0
        dfMeasure.to_csv(spath + meausureName + "_" + csvfile1.split("_")[-1], index=True)

def PvaluesCohensDvaluesEffectSizeLevelWinTieLoss(m, alpha=0.05, r=30):
    '''
    Calculate p-value by Wilcoxon_signed_rank_test(), d values and effect size level by Cohens_d(),
    win/tie/loss results by WinTieLoss().
    Note: All the files in the absolute path about csvfiles waiting for statistical test.
                  i.e., ============================
                        |  A,|   B,|   C,|    D,|...
                        ============================
                        |0.5,|0.79,| 0.7,| 0.43,|...
                        ============================
                  m: index of C; A,B & C are control methods for C.
    @param alpha: the values for statistical test of WIN/TIE/LOSS, default=0.05
    @param m: the index of column of data1, type: int
    @param r: the runtimes of one each dataset, i.e, the repeat values about one project
    @return: None
    '''

    pwd = os.getcwd()
    # print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    # resultspath = father_path + '/concactMeasureofDifferentMethodsFiles/'   # the first column is "testfile"
    # spath = father_path + '/StatsResultsLatexes/'
    # spath1 = father_path + '/StatsResultsCsvFiles/'

    resultspath = father_path + '/paperResults/concactCsv/'
    spath = father_path + '/paperResults/StatsResultsLatexes/'
    spath1 = father_path + '/paperResults/StatsResultsCsvFiles/'

    if not os.path.exists(spath):
        os.mkdir(spath)
    if not os.path.exists(spath1):
        os.mkdir(spath1)

    filelist = os.listdir(resultspath)
    for f in range(len(filelist)):
        df = pd.read_csv(resultspath + filelist[f])
        wintieloss = []
        df_stas_all = pd.DataFrame()
        columns_wtl = []
        d1 = df.iloc[:, m].astype(float)   # keep float data
        # for j in range(0, len(df.columns.values), 1):
        for j in range(7, len(df.columns.values), 1):
            if df.columns.values[j] != "testfile" and j != m:   # the first column is "testfile"
                d2 = df.iloc[:, j].astype(float)  # keep float data
                print("*******************win-tie-loss %s: %d/%d***************" % ("method", j, (len(df.columns)-1)))
                wtl = WinTieLoss(d1, d2, alpha, r)
                wintieloss.append(wtl)

                tempcolumn = df.columns.values[j]
                if ".csv" in tempcolumn:
                    tempcolumn = tempcolumn.rstrip(".csv")
                columns_wtl.append(tempcolumn)

                pvalues = []
                dvalues = []
                effectsizelevel = []
                # for i in range(0, len(d1), r):
                for i in range(90, len(d1), r):
                    data1 = df.iloc[i:i + r, m].astype(float)  # keep float data
                    data2 = df.iloc[i:i + r, j].astype(float)  # keep float data

                    print("*******************Wilcoxon_signed_rank_test: %d/%d***************" % (i, len(d1)))
                    stas, p = Wilcoxon_signed_rank_test(data1, data2)
                    pvalues.append("{:.2f}".format(p))

                    print("*******************Cohens_d & Effect size level: %d/%d***************" % (i, len(d1)))
                    d, esl = Cohens_d(data1, data2)
                    # d.map(lambda x: ('%.2f') % x)
                    dvalues.append("{:.2f}".format(d))  # round(d, 2)
                    effectsizelevel.append(esl)

                values = np.array([pvalues, dvalues, effectsizelevel]).T
                columnname = df.columns.values[j]
                if ".csv" in columnname:
                    columnname = columnname.rstrip(".csv")
                columns = [columnname + '-p',
                           columnname + "-d", columnname + "-esl"]

                df_stas = pd.DataFrame(values, columns=columns)
                df_stas_all = pd.concat([df_stas_all, df_stas], axis=1)

        indexname = []
        for t in range(0, len(df["testfile"].values), r):
            tempindexname = df["testfile"].values[t]
            if ".csv" in tempindexname:
                tempindexname = tempindexname.rstrip('.csv')
            indexname.append(tempindexname)

        df_stas_all.index = indexname
        df_stas_all.to_csv(spath1 + 'stas_' + filelist[f])
        df_stas_all.to_latex(spath + 'stas_' + filelist[f])
        # print(columns_wtl)
        df_wtl = pd.DataFrame(np.array(wintieloss), index=columns_wtl).T
        df_wtl.to_latex(spath + 'wtl_' + filelist[f])
        df_wtl.to_csv(spath1 + 'wtl_' + filelist[f])

def Csv2Latex():
    pwd = os.getcwd()
    # print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    resultspath = father_path + '/MethodsMeanStdMedianFiles/'   # the first column is "Unnamed: 0"
    spath = father_path + '/MethodsMeanStdMedianLatexes/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    filelist = os.listdir(resultspath)
    for f in range(len(filelist)):
        if ".csv" in filelist[f] and "_MeanStd_" not in filelist[f]:
            df = pd.read_csv(resultspath + filelist[f])
            index = df.iloc[:, 0].values
            csvname = filelist[f]
            if "_Mean_" or "_Median_" in csvname:
                format = (lambda x:'%.2f'%(x*100))  # value * 100 and keep 2 demicals
                df = df.iloc[:, 1:]  # the first column is "Unnamed: 0"
                df = df.applymap(format)
                # df = df.style.format("{:.2%}")
            elif "_Std_" in csvname:
                df = df.iloc[:, 1:]  # the first column is "Unnamed: 0"
                format = (lambda x:('%.2f')%float(x))  # .2%
                df = df.applymap(format)
                # df = df.style.format("{:.2f}")
            else:
                pass
            df.index = index
            df.to_latex(spath + 'Latex_' + filelist[f], index=True)

def judgeFirstColumn():
    """
    tempfunction: to judge the first column name is AUC or others.
    @return:
    """
    pwd = os.getcwd()
    # print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    path1 = father_path + '/results/'
    path2 = father_path + '/MethodsMeanStdMedianFiles/'
    path3 = father_path + '/MeanStdMedianFiles/'
    path4 = father_path + '/concactMeasureofDifferentMethodsFiles/'
    csvfile1 = os.listdir(path1)[0]
    csvfile2 = os.listdir(path2)[0]
    csvfile3 = os.listdir(path3)[0]
    csvfile4 = os.listdir(path4)[0]
    df1 = pd.read_csv(path1 + csvfile1)
    columnnames1 = df1.columns.values

    df2 = pd.read_csv(path2 + csvfile2)
    columnnames2 = df2.columns.values

    df3 = pd.read_csv(path3 + csvfile3)
    columnnames3 = df3.columns.values

    df4 = pd.read_csv(path4 + csvfile4)
    columnnames4 = df4.columns.values

    print(columnnames1,"\n" ,columnnames2,"\n",columnnames3,"\n",columnnames4)
    # path5 = father_path + '/concactMeasureofDifferentMethodsFiles/'
    # path6 = father_path + '/concactMeasureofDifferentMethodsFiles/'
    return True

def DirectlyCsv2Latex():
    """
    Just for X_MeanStd_Y.csv
    @return:
    """
    pwd = os.getcwd()
    # print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    resultspath = father_path + '/paperResults/csv/'  # the first column is "Unnamed: 0"
    spath = father_path + '/paperResults/latex/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    filelist = os.listdir(resultspath)
    for f in range(len(filelist)):
        if ".csv" in filelist[f]:
            df = pd.read_csv(resultspath + filelist[f])
            index = df.iloc[:, 0].values
            df = df.iloc[:, 1:]  # the first column is "Unnamed: 0"
            df.index = index
            df.to_latex(spath + 'Latex_' + filelist[f], index=True)

if __name__ == '__main__':
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    starttime = time.clock()  # 计时开始

    # iterable = [1, 2, 3, 4, 5, 6]
    # pool = multiprocessing.Pool()
    # cores = multiprocessing.cpu_count()
    # # print('cores:', cores)
    # func = partial(somefunc)
    # pool.map(func, iterable)
    # pool.close()
    # pool.join()

    # stepMethodMean()
    # concactDifferentDatasetsFiles()

    for measureindex in range(0, 9):
        MeanMedianStd(measureindex, learnernum=3)
    # Csv2Latex()

    # for measureindex in range(1, 10):
    #     concactMeasureofDifferentMethodsFiles(methodnum=8, learnernum=3, columnindex=measureindex)
    #
    # PvaluesCohensDvaluesEffectSizeLevelWinTieLoss(m=10, alpha=0.05, r=30)

    # judgeFirstColumn()
    # DirectlyCsv2Latex()
    endtime = time.clock()  # 计时结束

    totaltime = endtime - starttime
    print('程序结束运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序共运行时间为:', totaltime)




