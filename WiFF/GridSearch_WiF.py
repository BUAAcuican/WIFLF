#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
Main:
============================================================================
@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
@2020/10/20
============================================================================
"""
# print(__doc__)
from __future__ import print_function

import pandas as pd
import numpy as np
import time
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model

from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Process_Data, Random_Stratified_Sample_fraction, Standard_Features


def WiF_O2O(fpath, mode, clf_index, runtimes, spath):
    '''
    WiF Filter
    :param fpath:
    :param mode:
    :param clf_index:
    :param runtimes:
    :param spath:
    :return:
    '''

    preprocess_mode = mode[0]
    train_mode = mode[1]
    save_file_name = mode[2]
    iForest_parameters = mode[3]
    df_r_measures = pd.DataFrame()  # the measures of all files in runtimes

    classifiername = []
    for r in range(runtimes):
        if train_mode == 'O2O_CPDP':
            # df_file_measures, classifiername = O2O_CPDP(fpath, preprocess_mode, iForest_parameters, clf_index, r)
            file_list = os.listdir(fpath)
            df_file_measures = pd.DataFrame()  # the measures of all files in each runtime
            classifiername = []
            Sample_tr0 = NumericStringLabel2BinaryLabel(fpath + '\\' + file_list[0])
            column_name = Sample_tr0.columns.values  # the column name of the data
            for file_te in file_list:
                start_time = time.time()
                Address_te = fpath + '\\' + file_te
                Samples_te = NumericStringLabel2BinaryLabel(Address_te)  # DataFrame
                data = Samples_te.values  # DataFrame2Array
                X = data[:, : -1]  # test features
                y = data[:, -1]  # test labels
                df_tr_measures = pd.DataFrame()  # 每个测试集对应的训练集的所有结果
                for file_tr in file_list:
                    if file_tr != file_te:
                        # print(file_te, file_tr)
                        X_test = X
                        y_test = y
                        Address_tr = fpath + '\\' + file_tr
                        Samples_tr = NumericStringLabel2BinaryLabel(Address_tr)  # original train data, DataFrame
                        Samples_tr.columns = column_name.tolist()  # 批量更改列名
                        # Samples_tr.to_csv(f2, index=None, columns=None)  # 将类标签二元化的数据保存，保留列名，不增加行索引
                        # random sample 90% negative samples and 90% positive samples
                        string = 'bug'
                        Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                            = Random_Stratified_Sample_fraction(Samples_tr, string, r=r)
                        Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                        X_train_new, y_train_new, X_test_new, y_test_new = Process_Data(Sample_tr[:, :-1],
                                                                                        Sample_tr[:, -1],
                                                                                        X_test, y_test, preprocess_mode,
                                                                                        iForest_parameters, r)
                        # build and evaluation classification models
                        modelname, classifier = Selection_Classifications(clf_index, r)  # select classifier
                        classifiername.append(modelname)
                        measures = Build_Evaluation_Classification_Model(classifier, X_train_new, y_train_new,
                                                                         X_test_new,
                                                                         y_test_new)  # build and evaluate models
                        end_time = time.time()
                        run_time = end_time - start_time
                        measures.update({'train_len_before': len(Sample_tr),
                                         'train_len_after': len(X_train_new),
                                         'test_len': len(X_test_new),
                                         'runtime': run_time,
                                         'clf': clf_index,
                                         'testfile': file_te,
                                         'trainfile': file_tr})
                        df_o2ocp_measures = pd.DataFrame(measures, index=[r])
                        # print('df_o2ocp_measures:\n', df_o2ocp_measures)
                        df_tr_measures = pd.concat([df_tr_measures, df_o2ocp_measures],
                                                   axis=0, sort=False, ignore_index=False)
                # print('\n df_tr_measures:\n', df_tr_measures)
                df_file_measures = pd.concat([df_file_measures, df_tr_measures],
                                             axis=0, sort=False, ignore_index=False)

        else:
            raise ValueError('train_mode is wrong, please check...')

        df_file_measures['runtimes'] = r + 1

        # print('df_file_measures:\n', df_file_measures)
        # print('所有文件运行一次的结果为：\n', df_file_measures)
        df_r_measures = pd.concat([df_r_measures, df_file_measures], axis=0,
                                  sort=False, ignore_index=False)  # the measures of all files in runtimes
        # df_r_measures['testfile'] = file_list
        # print('df_r_measures1:\n', df_r_measures)
        method_name = mode[1] + '_' + mode[0][3]  # scenarios + filter method
        print('WiF:')
        print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))

    modelname = np.unique(classifiername)
    print('modelname:', modelname)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    if not os.path.exists(spath):
        os.mkdir(spath)

    df_r_measures.to_csv(pathname)
    # print('df_r_measures:\n', df_r_measures)

    return df_r_measures

def GridSearch_WiF(fpath, spath):
    print('###########DO GridSearch for WiF################')
    # default # subsample size: 256; Tree height：8; Number of trees: 100
    num_subsample = [16, 32, 64, 128, 256]  # phi  , 512
    num_tree = [10, 30, 50, 70, 90, 100]  # t , 110
    anomaly_rate = [0, 0.1, 0.2, 0.3, 0.4, 0.5]   # anomaly_rate = np.linspace(0., 0.5, num=8, retstep=True)  # alpha,闭区间生成8个等间隔数
    # max_depth  =  np.range(2, 15, 1)


    # grid search start
    best_score = 0
    best_score_auc = 0
    best_score_mcc = 0
    best_score_gm2 = 0
    best_score_gm = 0
    best_score_bal = 0
    measures = {}
    count = 0
    df_para_measures = pd.DataFrame()
    for t in num_tree:
        for phi in num_subsample:
            for alpha in anomaly_rate:
                count += 1  # 组合情况
                # 对于每种参数可能的组合，进行一次训练
                iForest_parameters = [t, phi, 255, 'rate', 0.5, alpha]  # [itree_num, subsamples, hlim, mode, s0, alpha]
                mode = [[1, 1, 1, 'Weight_iForest', 'zscore', 'smote'], 'O2O_CPDP', 'WiF', iForest_parameters]
                foldername = mode[1] + '_' + mode[0][3]

                # 使用NB做分类器
                df_r_measures = WiF_O2O(fpath=fpath, mode=mode, clf_index=0, runtimes=30, spath=spath + '\\' + foldername)
                measures.update({'t': t,
                                 'phi': phi,
                                 'alpha': alpha,
                                 'Balance': df_r_measures['Balance'].mean(),
                                 'MCC': df_r_measures['MCC'].mean(),
                                 'G-Mean2': df_r_measures['G-Mean2'].mean(),
                                 'G_Measure': df_r_measures['G-Measure'].mean(),
                                 'AUC': df_r_measures['AUC'].mean()})
                df_para_measure = pd.DataFrame(measures, index=[count])
                # para_results = []
                # print("measure1:", df_para_measure) # 每次参数的每次结果
                df_para_measures = pd.concat([df_para_measures, df_para_measure], axis=0, sort=False,
                                             ignore_index=False)

                score = df_r_measures['Balance'].mean()
                # 找到表现最好的参数
                if score > best_score:
                    best_score = score
                    best_parameters = {'t': t, "phi": phi, "alpha": alpha}

                score_bal = df_r_measures['Balance'].mean()
                score_auc = df_r_measures['AUC'].mean()
                score_gm2 = df_r_measures['G-Mean2'].mean()
                score_gm = df_r_measures['G-Measure'].mean()
                score_mcc = df_r_measures['MCC'].mean()

                if score_bal > best_score_bal:
                    best_score_bal = score_bal
                    best_para_bal = {'t': t, "phi": phi, "alpha": alpha}
                if score_auc > best_score_auc:
                    best_score_auc = score_auc
                    best_para_auc = {'t': t, "phi": phi, "alpha": alpha}
                if score_mcc > best_score_mcc:
                    best_score_mcc = score_mcc
                    best_para_mcc = {'t': t, "phi": phi, "alpha": alpha}
                if score_gm2 > best_score_gm2:
                    best_score_gm2 = score_gm2
                    best_para_gm2 = {'t': t, "phi": phi, "alpha": alpha}
                if score_gm > best_score_gm:
                    best_score_gm = score_gm
                    best_para_gm = {'t': t, "phi": phi, "alpha": alpha}

    print('Best socre:{:.2f}'.format(best_score))
    print('Best parameters:{}'.format(best_parameters))

    print('Best socre_bal:{:.2f}'.format(best_score_bal))
    print('Best para_bal:{}'.format(best_para_bal))

    print('Best socre_auc:{:.2f}'.format(best_score_auc))
    print('Best para_auc:{}'.format(best_para_auc))

    print('Best socre_mcc:{:.2f}'.format(best_score_mcc))
    print('Best para_mcc:{}'.format(best_para_mcc))


    print('Best socre_gm2:{:.2f}'.format(best_score_gm2))
    print('Best para_gm2:{}'.format(best_para_gm2))

    print('Best socre_gm:{:.2f}'.format(best_score_gm))
    print('Best para_gm:{}'.format(best_para_gm))

    # print("measure2:", df_para_measures)  # 所有不同参数组合的所有结果
    pathname = spath + '\\' + 'para_results.xlsx'
    if not os.path.exists(spath):
        os.mkdir(spath)

    df_para_measures.to_csv(pathname)  # save all results of different parameter combinations

def Draw3DFigures(fpath, spath):

    fig = plt.figure()
    ax = plt.axes(projection='3d')  # 利用关键字定义坐标轴
    # ax  = Axes3D(fig)  # 定义图像和三位格式坐标轴
    # ax = fig.add_subplot(111, projection='3d')  # 这种方法也可以画多个子图

    para_Data = pd.read_excel(fpath + '\\' + r't=64.xlsx')
    xx = para_Data['alpha'].values
    yy = para_Data['phi'].values

    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$\mathrm{\\phi} $')

    measure_list = ['Balance', 'G_Measure', 'G-Mean2', 'MCC', 'AUC']

    for str in measure_list:
        zz = para_Data[str].values
        ax.set_zlabel(str)
        ax.plot_trisurf(xx, yy, zz, linewidth=0.2, antialiased=False, cmap='rainbow')
        plt.savefig(spath + '\\' + str + ".png")
        plt.show()
    # zz = para_Data['Balance'].values
    # zz = para_Data['G_Measure'].values


    # ax.set_zlabel('Balance')
    # ax.set_zlabel('G-Measure')
    # ax.set_zlabel('G-Mean')
    # ax.scatter(xx, yy, zz)


    # ax.plot_surface(xx, yy, zz, cmap='rainbow') # Plot the surface, zz-2d
    # ax.plot_wireframe(xx, yy, zz, rstride=30, cstride=10) # Plot a basic wireframe, zz-2d
    # plt.savefig(spath + '\\' + "Balance.png")




if __name__ == '__main__':
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # starttime = time.clock()  # 计时开始
    starttime = time.perf_counter()

    # example
    fpath = r'D:\PycharmProjects\cuican\data2020\Example\data'
    spath = r'D:\PycharmProjects\cuican\data2020\Example'
    # GridSearch_WiF(fpath, spath)
    # help("matplotlib")
    Draw3DFigures(spath, spath)
    # AEEEM
    # fpath_a = r'D:\PycharmProjects\cuican\data2020\AEEEM\data'
    # spath_a = r'D:\PycharmProjects\cuican\data2020\AEEEM'

    # CK
    # fpath_e = r'D:\PycharmProjects\cuican\data2020\CK\earlier'
    # spath_e = r'D:\PycharmProjects\cuican\data2020\CK'


    # CK2
    # fpath_n = r'D:\PycharmProjects\cuican\data2020\CK2\newer'
    # spath_n = r'D:\PycharmProjects\cuican\data2020\CK2'


    # Relink
    # fpath_r = r'D:\PycharmProjects\cuican\data2020\Relink\data'
    # spath_r = r'D:\PycharmProjects\cuican\data2020\Relink'

    endtime = time.perf_counter()
    totaltime = endtime - starttime
    print('程序结束运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序共运行时间为:', totaltime)


