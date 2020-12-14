#!/usr/bin/env python
# -*- coding: UTF-8 -*-


"""
==============================================================
Bellwether
==============================================================
The code is about Bellwether for Bianary classification.

@ author: cuican
@ email: cuican5100923@163.com
==============================================================
"""
# print(__doc__)

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
import time
import copy
import sys
import multiprocessing
from functools import partial
from collections import Counter
from sklearn import preprocessing, model_selection, metrics,\
    linear_model, naive_bayes, tree, neighbors, ensemble, svm, neural_network

from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Standard_Features  # Process_Data,
from Performance import performance

# from TNB import TNB
# from TCAPlus import TCAplus




# Given N data sets, we find which one produces the best predicions on all the others

# This “bellwether” data set is then used for all subsequent predictions (or, until such time as its predictions
# start failing- at which point it is wise to seek another bellwether.


def weight_training(training_instance, test_instance):
    """
    test_instance：raw data，type：Dataframe
    training_instance：raw data，type：Dataframe
    target: normalization and delete nan columns in test and training data, return processed data.
    return：new_train[columns]：processed data，type：Dataframe
           new_test[columns]：processed data，type：Dataframe
    """
    # X_train = training_instance.iloc[:, :-1]
    # y_train = training_instance.iloc[:, -1]
    # X_test = test_instance.iloc[:, :-1]
    # y_test = test_instance.iloc[:, :-1]
    # X_train_new = (X_train - X_test.mean()) / X_test.std()  # normalization
    # # training_instance.iloc[:, :-1] = X_train_new
    # new_train = np.c_[X_train_new, y_train]
    # loc = np.where(np.isnan(new_train))
    # new_train.index[np.where(np.isnan(new_train))[0]]
    # new_train.dropna(axis=1, inplace=True)  # 将所有存在空的列删除
    # head = training_instance.columns
    # tgt = new_train.columns
    # X_test_new = (X_test - X_test.mean()) / X_test.std()  # normalization

    head = training_instance.columns
    test_instance.columns = head  # keep the columns name same
    new_train = training_instance[head[:-1]]
    new_train = (new_train - test_instance[head[:-1]].mean()) / test_instance[head[:-1]].std()  # normalization
    new_train[head[-1]] = training_instance[head[-1]]

    new_train.dropna(axis=1, inplace=True)  # 将所有存在空的列删除
    tgt = new_train.columns
    new_test = (test_instance[tgt[:-1]] - test_instance[tgt[:-1]].mean()) / (
        test_instance[tgt[:-1]].std())

    new_test[tgt[-1]] = test_instance[tgt[-1]]
    new_test.dropna(axis=1, inplace=True)
    columns = list(set(tgt[:-1]).intersection(new_test.columns[:-1])) + [tgt[-1]]
    return new_train[columns], new_test[columns]



def SMOTE_Data(df_data, r):
    '''

    :param df_data: Type:DataFrame, the last column is binary (0 or 1).
    :param r: Random State, r>=0
    :return: Resampled X,y
    '''

    X = df_data.iloc[:, :-1]
    y = df_data.iloc[:, -1]
    nondefct_num = len(y) - np.sum(y)
    defect_num = np.round((1 / 2) * nondefct_num)
    # print(Counter(y), nondefct_num, defect_num)
    if nondefct_num >= np.sum(y) and defect_num > np.sum(y):
        # SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本
        X_resampled_smote, y_resampled_smote = SMOTE(random_state=r + 1, ratio={1: int(defect_num)}).fit_sample(X, y)
        y_smote = sorted(Counter(y_resampled_smote).items())
        # print(y_smote)
        return X_resampled_smote, y_resampled_smote
    else:
        X_resampled_rus, y_resampled_rus = RandomUnderSampler(ratio={1: int(defect_num)}).fit_sample(X, y)
        y_rus = sorted(Counter(y_resampled_rus).items())
        # print(y_rus)
        return X_resampled_rus, y_resampled_rus

def train(data1, clf_index, r):
    """
    train train_data (ie. data1) using clf
    :param data1:
    :param clf_index
    :param r
    :return:
    """
    X_resampled, y_resampled = SMOTE_Data(data1, r)
    rf = ensemble.RandomForestClassifier(random_state=r + 1, n_estimators=100)
    lr = linear_model.LogisticRegression(random_state=r + 1, solver='liblinear', max_iter=100, multi_class='ovr')
    gnb = naive_bayes.GaussianNB(priors=None)
    clfs = [{'NB': gnb}, {'LR': lr}, {'RF': rf}] #, {'KNN': knn},{'SVMR': svmr}, {'MLP': mlp}, {'SVML': svml}, {'SVMP': svmp}  # {'DT': dt},

    classifier = clfs[clf_index]
    modelname = str(list(classifier.keys())[0])  # 分类器的名称
    # print('predictor:', modelname)
    classifier = list(classifier.values())[0]  # 分类器参数模型
    classifier.fit(X_resampled, y_resampled)

    return classifier

def Predict(classifier, data2):
    '''
    Predict for quality
    :param classifier:
    :param data2:
    :return:
    '''

    X_test = data2.iloc[:, :-1]
    y_test = data2.iloc[:, -1]
    # y_te_prepredict = classifier.predict(X_test)  # predict labels
    y_te_predict_proba = classifier.predict_proba(X_test)  # predict probability
    prob_pos = []  # the probability of predicting as positive
    prob_neg = list()  # the probability of predicting as negative
    for j in range(y_te_predict_proba.shape[0]):  # 每个类的概率, array类型
        if classifier.classes_[1] == 1:
            prob_pos.append(y_te_predict_proba[j][1])
            prob_neg.append(y_te_predict_proba[j][0])
        else:
            prob_pos.append(y_te_predict_proba[j][0])
            prob_neg.append(y_te_predict_proba[j][1])
    prob_pos = np.array(prob_pos)
    prob_neg = np.array(prob_neg)
    measures = performance(y_test, prob_pos)
    return measures

def Score(data1, data2, Runtimes, clf_index=2, measure = 'G-Measure'):
    '''
    Return accuracy of Prediction
    :param data1:
    :param data2:
    :param Runtimes:
    :return:
    '''
    data1_new, data2_new = weight_training(data1, data2)  # processing data1 & data2
    score_sum = 0
    for r in range(Runtimes):
        predictor = train(data1_new, clf_index, r=r)
        measures = Predict(predictor, data2_new)
        # print(measures)
        value = measures[measure]  # visit 'value' by 'key'
        if value != np.nan:
            score_sum += value
        else:
            pass
    value_mean = score_sum / Runtimes

    return value_mean

def discover(datasets, Runtimes, clf_index=2):
    '''
    Identify Bellwether Datasets
    :param datasets: type: list, each element includes absolute path of .csv,
                     e.g.'D:\PycharmProjects\cuican\data2020\Relink\data\Apache.csv'
    :return: Return data with best prediction score
    For example, 3 projects
    1 as data1;2&3 as data2, respectively. --> 1v2-score12,1v3--score13 --->mean1 = (score12+score13).mean ;
    2 as data1;1&3 as data2, respectively. --> 2v1-score21,2v3--score23 --->mean2 = (score21+score23).mean;
    3 as data1;1&2 as data2, respectively. --> 3v1-score31,3v2--score32 --->mean3 = (score31+score32).mean ;
    max(mean1, mean2, mean3)=x -->x project of(1,2,3) projects is a bellwether dataset.
    '''
    score_values = []  # 初始化每个data作为data1时得到的score均值
    for file1 in datasets:
        data1 = NumericStringLabel2BinaryLabel(file1)  # train data:Binary Label
        data1_score_values = []
        data1_score_mean = 0  # 初始化data1作为训练集，其他data分别作为测试集时的score均值
        for file2 in datasets:
            if file1 != file2:
                data2 = NumericStringLabel2BinaryLabel(file2)  # test data
                value_mean = Score(data1, data2, Runtimes, clf_index)
                data1_score_values.append(value_mean)
                data1_score_mean += value_mean
        data1_score_mean = data1_score_mean / (len(datasets) - 1)
        score_values.append(data1_score_mean)
        # print("train:", file1, data1_score_mean)
    dataset_index = score_values.index(max(score_values))
    bellwether_path = datasets[dataset_index]
    bellwether = NumericStringLabel2BinaryLabel(bellwether_path)
    return bellwether, bellwether_path

def discoverMore(datasets, Runtimes, clf_index=2):
    '''
    Identify Bellwether Datasets for more projects
    :param datasets: type: list, each element includes absolute path of .csv,
                     e.g.'D:\PycharmProjects\cuican\data2020\Relink\data\Apache.csv'
    :return: Return data with best prediction score
    For example, 3 projects
    1 as data1;2+3 as data2 --> 1v(2+3)--score1--->mean1 = (score1).mean ;
    2 as data1;1+3 as data2 --> 2v(1+3)--score2--->mean2 = (score2).mean;
    3 as data1;1+2 as data2 --> 3v(1+2)--score3 --->mean3 = (score3).mean ;
    max(mean1, mean2, mean3)=x -->x project of(1,2,3) projects is a bellwether dataset.
    '''
    score_values = []  # 初始化每个data作为data1时得到的score均值
    for file1 in datasets:
        data1 = NumericStringLabel2BinaryLabel(file1)  # train data:Binary Label
        head = data1.columns
        # data1_score_values = []
        # data1_score_mean = 0  # 初始化data1作为训练集，其他data分别作为测试集时的score均值
        data2 = pd.DataFrame()
        for file2 in datasets:
            if file1 != file2:
                data_Temp = NumericStringLabel2BinaryLabel(file2)  # test data, type: DataFrame
                data_Temp.columns = head  # keep the columns name same
                data2 = pd.concat([data2, data_Temp], sort=False)
        data1_score_mean = Score(data1, data2, Runtimes, clf_index)
        score_values.append(data1_score_mean)
        # print("train:", file1, data1_score_mean)
    dataset_index = score_values.index(max(score_values))
    bellwether_path = datasets[dataset_index]
    bellwether = NumericStringLabel2BinaryLabel(bellwether_path)  # type, dataframe
    return bellwether, bellwether_path

def old_learner(source, target, learnername, Runtimes, clf_index=2):
    '''
    Conduct Transfer learner, Using:
    1.TCA+; 2.TNB; 3. VCB; 4. Naive
    :param source:
    :param target:
    :param learnername: transfer learner name, type: string
    :param r:
    :return: the Runmtimes results that source train model to predict test, type: DataFrame
    '''
    df_te_measures = pd.DataFrame()  # 每个测试集对应的训练集的所有结果
    if learnername == 'TCA+':
        print('###########DO TCAPlus:O2O################')
        source_tcaplus, target_tcaplus = TCAplus(source, target)
        for r in range(Runtimes):
            predictor = train(source_tcaplus, clf_index, r=r)
            measures = Predict(predictor, target_tcaplus)
            df_te_once_measures = pd.DataFrame(measures, index=[r])
            # print('df_te_once_measures:\n', df_te_once_measures)
            df_te_measures = pd.concat([df_te_measures, df_te_once_measures],
                                       axis=0, sort=False, ignore_index=False)
        print('\n df_te_measures:\n', df_te_measures)

    elif learnername == 'TNB':
        for r in range(Runtimes):
            measures = TNB(source, target, isDiscrete=True)
            df_te_once_measures = pd.DataFrame(measures, index=[r])
            # print('df_te_once_measures:\n', df_te_once_measures)
            df_te_measures = pd.concat([df_te_measures, df_te_once_measures],
                                       axis=0, sort=False, ignore_index=False)

    elif learnername == 'VCB':
        pass
        # return True

    elif learnername == 'Naive':
        for r in range(Runtimes):
            predictor = train(source, clf_index=2, r=r)
            measures = Predict(predictor, target)
            print("measures", measures)
            df_te_once_measures = pd.DataFrame(measures, index=[r])
            # print('df_te_once_measures:\n', df_te_once_measures)
            df_te_measures = pd.concat([df_te_measures, df_te_once_measures],
                                       axis=0, sort=False, ignore_index=False)
    return df_te_measures

def transfer(fpath, learnername, spath, Runtimes, clf_index=2):
    '''
    Transfer Learning with Bellwether Dataset
    :param datasets:
    :return:
    '''
    filelist = os.listdir(fpath)
    df_file_measures = pd.DataFrame()   # the measures of all files in each runtime
    for targetfile in filelist:
        target = pd.read_csv(fpath + '\\' + targetfile)  # target data
        datasets = []
        for datafile in filelist:
            if datafile != targetfile:
                datasets.append(fpath + '\\' + datafile)
        bellwether, bellwether_path = discover(datasets, Runtimes, clf_index)  # bellwether data
        print('target:', targetfile, "bellwether:", bellwether_path)
        # learnername = 'Naive'
        df_te_measures = old_learner(bellwether, target, learnername, Runtimes)
        df_file_measures = pd.concat([df_file_measures, df_te_measures],
                                     axis=0, sort=False, ignore_index=False)
    print('所有文件运行%d的结果df_file_measures为：\n', (Runtimes, df_file_measures))
    # print(measures)
    return df_file_measures

def learner(source, target, learnername, clf_index, r):
    '''
    Conduct Transfer learner, Using:
    1.TCA+; 2.TNB; 3. VCB; 4. Naive
    :param source:
    :param target:
    :param r: random state
    :return: the one result that source train model to predict test, type: dict
    '''
    measures = {}  # 测试集对应的训练集的结果
    if learnername == 'TCA+':
        print('###########DO TCAPlus:O2O################')
        source_tcaplus, target_tcaplus = TCAplus(source, target)
        predictor = train(source_tcaplus, clf_index, r=r)
        measures = Predict(predictor, target_tcaplus)
        # print("measures", measures)
    elif learnername == 'TNB':
        measures = TNB(source, target, isDiscrete=True)
        # print("measures", measures)
    elif learnername == 'VCB':
        pass
            # return True

    elif learnername == 'Naive':
        predictor = train(source, clf_index, r=r)
        measures = Predict(predictor, target)
        # print("measures", measures)

    return measures

def apply_learner(datasets, learner):
        '''
        Apply transfer learner
        :param datasets:
        :param learner:
        :return:
        '''
        model = learner()
        return True


def Bellwether_main(learnername, mode, clf_index, runtimes):

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

                    # /* step1: data preprocessing */
                    # random sample 90% negative samples and 90% positive samples
                    Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug', r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    source = np.c_[X_train_zscore, y_train]
                    target = np.c_[X_test_zscore, y_test]

                    # /* step2: Bellwether filtering processing */
                    # *******************Bellwether*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    source = pd.DataFrame(source)
                    source.columns = column_name.tolist()  # 批量更改列名
                    target = pd.DataFrame(target)  # note: type transfer
                    target.columns = column_name.tolist()  # 批量更改列名
                    measures_bellwether = learner(source, target, learnername, clf_index, r)  # learner

                    # /* step3: Model building and evaluation */
                    # Train，build and evaluate model: model requires the label must beong to {0, 1}.

                    modeldict = {0: "NB", 1: "LR", 2: "RF"}
                    modelname_bellwether = modeldict[clf_index]
                    classifiername.append(modelname_bellwether)
                    # print("modelname:", modelname)
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_bellwether.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(source), 'test_len': len(X_test_zscore),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_bellwether,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_bellwether, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False, ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False,
                                         ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures





def _test_SMOTE_Data():
    fpath = r'D:\PycharmProjects\cuican\data2020\Relink\data'
    file = fpath + "\\" + 'Apache.csv'
    df_data = NumericStringLabel2BinaryLabel(file)
    X_resampled, y_resampled = SMOTE_Data(df_data, r=0)
    print(X_resampled, y_resampled)

def _test_discover():
    fpath = r'D:\PycharmProjects\cuican\data2020\CK\earlier'
    file1 = fpath + "\\" + 'ant-1.3.csv'
    file2 = fpath + "\\" + 'arc-1.csv'
    file3 = fpath + "\\" + 'ivy-1.4.csv'
    datasets = [file1, file2, file3]
    bellwether, bellwether_path = discover(datasets, Runtimes=30)
    print(bellwether_path)

if __name__ == '__main__':
    # _test_SMOTE_Data()
    # _test_discover()
    # fpath = r'D:\PycharmProjects\cuican\data2020\Relink\data'
    # transfer(fpath, 'Naive', fpath, Runtimes=30)

    mode = ['null', 'M2O_CPDP', 'eg_Bellwether']
    foldername = mode[1] + '_' + mode[2]
    for i in range(1, 3, 1):
        Bellwether_main('Naive', mode, i, runtimes=3)

