#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
TNB:transfer naive Bayes
============================================================================
@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
@2019/12/23-2019/12/27

============================================================================
"""
# print(__doc__)

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import pandas as pd
import numpy as np
import math
import heapq
import random
import copy
import time
import os
from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Log_transformation, Standard_Features  # Process_Data,
from Performance import performance
from collections import Counter

def get_maxmin_value(martix):
    '''
    calculate the minimum and maximum values of each column of a martix.
    :param martix: 2-D array
    :return: max_columns: list contains max values of each column
             min_columns: list contains min values of each column
    '''

    max_columns = []
    min_columns = []
    for j in range(martix.shape[1]):  # columns number
        # print()
        max_column = np.max(martix[:, j])
        max_columns.append(max_column)

        min_column = np.min(martix[:, j])
        min_columns.append(min_column)

    return max_columns, min_columns

def class_entropy(classes):
    temp_dict = dict(Counter(classes))
    class_count = list(temp_dict.values())  # the number of each class
    class_unique = list(temp_dict.keys())  # the label
    class_number = len(class_count)  # number of unique classes?
    # class_number = len(classes)
    P_cs = []
    for i in range(len(class_unique)):
        P_cs_temp = class_count[i] / len(classes)
        if P_cs_temp == 0:
            P_cs_temp = 0.000001  # avoid the log2(0)
        else:
            pass
        P_cs.append(P_cs_temp)
    s_ent = -np.sum(P_cs * np.log2(P_cs))  # class entropy
    return s_ent, class_number

def fayyadIrani(points, classes, intervals=40):
    '''
    NOTES:
    This is a pretty neat implementation of information theory.In my empirical
    observation, this is by far the fastest discretizing metric I've come across.
    In addition, it seems to be quite accurate.
    CITATIONS
    code implements Fayyad and Irani's discretization algorithms as described
    in Kohavi, Dougherty - 'Supervised and Unsupervised Discretization of Continuous Features ' and
    ' Li, Wong ' - Emerging Patterns and Gene Expression data = >
    http://hc.ims.u-tokyo.ac.jp/JSBi/journal/GIW01/GIW01F01/GIW01F01.html.
    Lawrence David - 2003. lad2002 @ columbia.edu
    Tom Macura - 2005. tm289 @ cam.ac.uk(modified for inclusion in OME)
    warning off MATLAB: colon:operandsNotRealScalar;  # shutup

    :param points: the 1 - dimensional values of each instance
    :param classes: class numbers {0, 1} of all points, namely class labels of all points.
    :param intervals: how much you want to break up each smaller section. OPTIONAL.The default is 40.
    :return: walls - vector containing all "walls"
    '''

    walls = []
    # points = double(points)  # sometimes, data comes in as 'single'
    # classes = double(classes)

    # Erect walls between min and max to divide distance into interval intervals
    if np.max(points) != np.min(points):
        step = (np.max(points) - np.min(points)) / intervals
        bin_walls = list(np.arange(np.min(points), np.max(points),
                                   step))  # A vector, begin from min(points), end at max(points), the gap is '(max(points)-min(points))/intervals'

        if bin_walls != []:  # stop if you split into space with no points
            bin_num = len(bin_walls)  # number of bin_walls: cut points
            N = len(points)  # number of samples
            # compute class entropy and number of unique classes
            s_ent, class_number = class_entropy(classes)  # get class entropies by calling self-defined function
            class_info_ent = []  # class_info_ent =  np.zeros((bin_num, 1))
            #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #   #  #
            # looking for best way to split points into two parts
            #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  ##  #  #
            s1_ent = np.zeros((bin_num, 1))
            s1_k = np.zeros((bin_num, 1))
            s2_ent = np.zeros((bin_num, 1))
            s2_k = np.zeros((bin_num, 1))
            # print('bin_num:', bin_num)
            for i in range(bin_num):
                #  Divid samles of a feature into two parts
                temp = (points > bin_walls[
                    i]) + 1  # logical value plus a numeric value becomes a numeric value simple discretization; two cases: <= & >

                # Corrsponding labels of each part
                s1 = classes[
                    [temp == 1]]  # '[temp==1]' denotes the index of elements which are not larger than bin_walls(i)
                s2 = classes[
                    [temp == 2]]  # '[temp==2]' denotes the index of elements which are larger than bin_walls(i)

                # Number of elements in each part
                s1_size = len(s1)
                s2_size = len(s2)

                # Class entropy
                s1_ent[i], s1_k[i] = class_entropy(s1)  # get class entropies by calling self-defined function
                s2_ent[i], s2_k[i] = class_entropy(s2)  # get class entropies by calling self-defined function

                class_info_ent.append((s1_size / N) * s1_ent[i] + (s2_size / N) * s2_ent[
                    i])  # (P71: formula (1))--want to minimize this baby

            # Definition by Fayyad and Irani 's discretization algorithms
            low_score = np.min(class_info_ent)  # the minimum value of the class information entropy
            high_score = np.min(class_info_ent)
            if low_score == high_score:
                lsp = class_info_ent.index(np.min(
                    class_info_ent))  # the index position of the minimum class information entropy, how to do if there are more than one index?
            else:
                lsp_all = [i for i, d in enumerate(class_info_ent) if d == low_score]
                if len(lsp_all) > 1:
                    print('lsp_all:', lsp_all)

            # print('class_info_ent:', class_info_ent, 'len(class_info_ent):', len(class_info_ent))
            # print('low_score:', low_score, 'lsp:', lsp)

            p1 = points[[points < bin_walls[lsp]]]  # number games...
            p2 = points[[points > bin_walls[lsp]]]

            c1 = classes[[points < bin_walls[lsp]]]
            c2 = classes[[points > bin_walls[lsp]]]

            gain = s_ent - class_info_ent[lsp]  # do we have enough information gain?
            # print('class_number:', class_number)
            deltaATS = np.log2(3 ** class_number - 2) - (class_number * s_ent - s1_k[lsp] * s1_ent[lsp] - s2_k[lsp] * s2_ent[lsp])
            # print('deltaATS:', deltaATS)
            right_side = np.log2(N - 1) / N + deltaATS / N

            #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
            # descend .recursively  #
            #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
            if gain >= right_side:
                # FIXME intervals is hard - coded to 25. 
                intervals = 25
                bw = bin_walls[lsp]
                walls.append(bw)
                if len(p1) > 0:
                    fI1 = fayyadIrani(p1, c1, intervals)  # recursive partition
                    if len(fI1) > 0:
                        walls.extend(fI1)
                if len(p2) > 0:
                    fI2 = fayyadIrani(p2, c2, intervals)  # recursive partition
                    if len(fI2) > 0:
                        walls.extend(fI2)

            else:
                walls = []  # stop if you reach the stop condition
        else:
            walls = []  # if you're having no luck, just give up
    else:
        walls = []  # if you're having no luck, just give up

    # print(walls)
    return walls


def TNB(Source, Target, isDiscrete):
    '''
    TNB():
    implements transfer naive Bayes.
    Reference: P47-2012-Transfer learning for cross - company software defect prediction,Ying Ma
    Detailed explanation goes here:
    :param Source: [S1,S2,...,SN], each row of Si corresponds to a observation and the last column is label {0, 1}.
    :param Target: test data from target project, each row is a observation.Source and TargetData have same metrics, namely
    the number of columns is equal.
    :param isDiscrete: {0, 1} - 1 denotes that each metric is discrete, continuous otherwise.
    :return: measures, dict, performance measures: PD ,PF ,Precision ,F1 ,AUC ,Accuracy ,G_measure ,MCC, Balance...
    '''

    # Drop_Duplicate_Samples:(have finished in main() function)
    # Source = Drop_Duplicate_Samples(pd.DataFrame(Source))  # cannot change the number of the target
    Target = pd.DataFrame(Target)
    Source = pd.DataFrame(Source)
    # 防止数据集规范化后出现全零属性
    for i in range(len(Target.columns.values) - 1):  # each feature
        #  Target
        if np.max(Target.iloc[:, i]) == np.min(Target.iloc[:, i]) and np.max(Target.iloc[:, i]) == 0:
            Target.iloc[:, i] = np.tile(0.000001, (len(Target), 1))
        # Source
        if np.max(Source.iloc[:, i]) == np.min(Source.iloc[:, i]) and np.max(Source.iloc[:, i]) == 0:
            Source.iloc[:, i] = np.tile(0.000001, (len(Source), 1))  # [0.01] * len(Source)   #

    Source = Source.values
    Target = Target.values
    # print('Source:', Source, type(Source))

    # Log tranform all feature values, x -> ln(x) and 0 -> 1e-6
    Source_x = Log_transformation(Source[:, :-1], alpha=0.000001)  # 1e-6 = 0.000001
    Target_x = Log_transformation(Target[:, :-1], alpha=0.000001)
    Source = np.c_[Source_x, Source[:, -1]]
    Target = np.c_[Target_x, Target[:, -1]]
    # print('Source_log:', Source, 'Target_log:', Target)


    # maximun and minimum of each feature in target (3.2.1-(2)(3))
    Max, Min = get_maxmin_value(Target[:, :-1])  # the maximum/minimum value of each feature

    # Calculate similarity (3.2.1) and weight of each training instance based on target
    s = np.zeros((len(Source), 1))  # s - the similarity of each training instance
    w = np.zeros((len(Source), 1))  # w - the weight of each training instance
    for i in range(len(Source)):  # each source instance
        tem = 0
        for j in range(len(Max)):  # each feature
            if (Source[i, j] >= Min[j]) and (Source[i, j] <= Max[j]):
                tem = tem + 1
            else:
                tem = tem + 0
        s[i] = tem   # similarity for training data (3.2.1-(4))
        w[i] = s[i] / (Source.shape[1] - 1 - s[i] + 1) ** 2  # weighting for training data (3.2.3-(6))
    # print('similarity:', s)
    # print('weight:', w)
    #
    w = w / np.sum(w)  # normalize the weight (cc)

    # Calculate the prior probability of each class(i.e., positive class and negative class )
    label = Source[:, -1]  # the label of source
    nc = len(np.unique(label))
    # print('nc:', nc, 'n:', len(Source))
    wc_pos = 0
    wc_neg = 0
    for l in range(len(Source)):
        if label[l] == 1:
            wc_pos = wc_pos + w[l] * 1
            # print('wc_pos:', wc_pos)
        else:
            wc_neg = wc_neg + w[l] * 1
            # print('ws_neg:', wc_neg)
    # print('sum(w):', np.sum(w))
    pri_prob_neg = (wc_neg + 1) / (np.sum(w) + nc)  # the prior probability of each class (3.2.3-(7))
    pri_prob_pos = (wc_pos + 1) / (np.sum(w) + nc)  # the prior probability of each class (3.2.3-(7))
    # print('P_false:', pri_prob_neg, 'P_true:', pri_prob_pos)

    pred_label = np.zeros((len(Target), 1))
    pro_pos = np.zeros((len(Target), 1))
    pro_neg = np.zeros((len(Target), 1))
    pos_cond_met = np.zeros((Source.shape[1]-1, 1))
    neg_cond_met = np.zeros((Source.shape[1]-1, 1))
    for i in range(len(Target)):  # Each instance in Target
        met = Target[i, :]  # i - th instance in Target
        walls = []
        idx1 = []  # positive
        idx2 = []  # negative
        nj = 0
        for j in range(len(met)-1):  # Each feature
            wac_pos = 0
            wac_neg = 0
            wc_pos = 0
            wc_neg = 0
            if isDiscrete:  # 离散
                nj = len(np.unique(Source[:, j]))  # the total number of unique values of j - th metric in Source
                # print('j:', j+1, 'nj:', nj)
                for l in range(len(Source)):
                    if Source[l, j] == met[j] and label[l] == 1:
                        idx1.append(l)  # the total number of positive instances in source whose j - th metric value equals to met(j)
                        wac_pos = wac_pos + w[l] * 1 * 1
                        # print('wac_pos:', wac_pos)
                    elif Source[l, j] == met[j] and label[l] == 0:
                        idx2.append(l)  # the total number of negative instances in source whose j - th metric value equals to met(j)
                        wac_neg = wac_neg + w[l] * 1 * 1
                        # print('wac_neg:', wac_neg)

                for l in range(len(Source)):
                    if label[l] == 1:
                        wc_pos = wc_pos + w[l] * 1
                        # print('wc_pos:', wc_pos)
                    else:
                        wc_neg = wc_neg + w[l] * 1
                        # print('wc_neg:', wc_neg)

            else:  # 连续
                walls = fayyadIrani(Source[:, j], label)  # Call fayyadIrani() to generate walls.
                # print('walls:', walls)
                walls.append(np.min(Source[:, j]) + np.min(Source[:, j]) / 2)   # why - - min(Source(:, j)) + min(Source(:, j)) / 2, max(Source(:, j))?
                walls.append(np.max(Source[:, j]))
                walls.sort()  # walls = walls.sort()  return None
                nj = len(walls) - 1  # the number of intervals

                # [infimum, supremum] is the most nearest wall interval which includes met[j] in ideal condition.
                supremum = 0  # to be a global variable
                infimum = 0  # to be a global variable
                supre_index = []
                infi_index = []
                for m in range(len(walls)):
                    if np.round(walls[m], 6) >= np.round(met[j], 6):
                        supre_index.append(m)
                    elif np.round(walls[m], 6) <= np.round(met[j], 6):
                        infi_index.append(m)
                if len(supre_index) > 0:
                    supremum = walls[np.min(supre_index)]
                else:
                    supremum = np.max(walls)  # To avoid supremum or infimum is empty when met(j) is larger than max(walls) or met(j) is smaller than min(walls).
                if len(infi_index) > 0:
                    infimum = walls[np.max(infi_index)]
                else:
                    infimum = np.min(walls)
                # print('supre_index:', supre_index)
                # print(type(supremum), supremum.shape)

                for k in range(len(Source)):
                    if (Source[k, j] > infimum) and (Source[k, j] <= supremum) and (label[k] == 1):
                        idx1.append(i)  # the total number of positive instances which belong to the interval(infinum, supremum].
                        wac_pos = wac_pos + w[k] * 1 * 1
                    elif (Source[k, j] > infimum) and (Source[k, j] <= supremum) and (label[k] == 0):
                        idx2.append(i)    # the total number of negative instances which belong to the interval(infinum, supremum].
                        wac_neg = wac_neg + w[k] * 1 * 1

                for l in range(len(Source)):
                    if label[l] == 1:
                        wc_pos = wc_pos + w[l] * 1
                    else:
                        wc_neg = wc_neg + w[l] * 1

            # Calculate the class -conditionnal probability for j-th metric  (3.2.3-(8))
            # print('numerator_neg：', (wac_neg + 1), 'denominator_neg', (wc_neg + nj))
            # print('numerator_pos：', (wac_pos + 1), 'denominator_pos', (wc_pos + nj))
            pos_cond_met[j] = (wac_pos + 1) / (wc_pos + nj)
            neg_cond_met[j] = (wac_neg + 1) / (wc_neg + nj)
            # print('P_aj_c_false:', neg_cond_met[j], 'P_aj_c_true:', pos_cond_met[j])

        # print('P_ac_false:', neg_cond_met, 'P_ac_true:', pos_cond_met)  # class -conditionnal peobability

        # Calculate posterior probability
        deno = pri_prob_pos * np.prod(pos_cond_met) + pri_prob_neg * np.prod(neg_cond_met)  # prod([1, 2, 3]) = > 6
        # print('deno:', deno)
        if deno != 0:
            pro_pos[i] = pri_prob_pos * np.prod(pos_cond_met) / deno
            pro_neg[i] = pri_prob_neg * np.prod(neg_cond_met) / deno
        else:
            # raise ('ValueError! Check the pro values carefully!')
            pro_pos[i] = pri_prob_pos * np.prod(pos_cond_met) / 0.000001  # 正常情况不会执行该代码
            pro_neg[i] = pri_prob_neg * np.prod(neg_cond_met) / 0.000001  # 正常情况不会执行该代码

        # print('pro_pos[i]:', pro_pos[i],  'pro_neg[i]:', pro_neg[i])

        # Predicted labels
        if pro_pos[i] >= pro_neg[i]:
            pred_label[i] = 1
        else:
            pred_label[i] = 0
    # print('pro_pos', pro_pos,  'pro_neg:', pro_neg)
    meaures = performance(Target[:, -1], pro_pos)  # Call self - defined performance()
    # print('measures:', meaures)
    # return pred_label, meaures
    return meaures

def example_isDiscrete():
    Train_Data = np.array([[2, 1, 3, 0, 0],
                           [1, 2, 2, 0, 0],
                           [1, 3, 4, 0, 1]])
    x1 = Train_Data[0, :]
    x2 = Train_Data[1, :]
    x3 = Train_Data[2, :]

    Test_Data_unlabel = np.array([[2, 1, 3, 1, 0], [1, 2, 3, 1, 1]])

    u1 = Test_Data[0, :]
    u2 = Test_Data[1, :]

    # TNB(Train_Data, Test_Data, isDiscrete=True)  # 成功

def example_isContinuous():
    fpath = r'D:\PycharmProjects\cuican\functions\Weight_iForest\PROMISE_numeric\all'  # \example_o2ocp
    trname = r'camel-1.0.csv'
    tename = r'e-learning-1.csv'  # r'lucene-2.0.csv'
    source_x, source_y = DataFrame2Array(fpath + '\\' + trname)
    target_x, target_y = DataFrame2Array(fpath + '\\' + tename)
    source = np.c_[source_x, source_y]
    target = np.c_[target_x, target_y]
    TNB(source, target, isDiscrete=False)

def TNB_main(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    # datasets for RQ2
    datasets = [
        ['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv',
         'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv',
         'xalan-2.4.csv', 'xerces-init.csv'],
        ['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv', 'lucene-2.0.csv',
         'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv',
         'xalan-2.6.csv', 'xerces-1.3.csv'],
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
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug',
                                                            r=r)  # random sample 90% negative samples and 90% positive samples
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
                    # Train，build and evaluate model: model requires the label must beong to {0, 1}.
                    measures = TNB(source, target, isDiscrete=False)  # the drop step has been commented in TNB()
                    modelname = 'TNB'
                    classifiername.append(modelname)
                    # print("modelname:", modelname)

                    end_time = time.time()
                    run_time = end_time - start_time
                    measures.update({'train_len_before': len(Sample_tr),
                                     'train_len_after': len(source),
                                     'test_len': len(target),
                                     'runtime': run_time,
                                     'clf': clf_index,
                                     'testfile': file_te,
                                     'trainfile': 'More1'})
                    df_m2ocp_measures = pd.DataFrame(measures, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_file_measures = pd.concat([df_file_measures, df_m2ocp_measures], axis=0, sort=False,
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
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    starttime = time.clock()  # 计时开始

    # example_isDiscrete()
    # example_isContinuous()
    # fayyadIrani(points, classes, intervals=40)

    mode = ['null', 'M2O_CPDP', 'eg_TNB']
    TNB_main(mode=mode, clf_index=0, runtimes=2)

    endtime = time.clock()  # 计时结束
    totaltime = endtime - starttime
    print('程序结束运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序共运行时间为:', totaltime)





