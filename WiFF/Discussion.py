#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
Discussion in Paper:
Weighted Isolation Forest Filter for Instance-based Cross-Project Defect Prediction
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

from WiFF import WiFF_main



def GridSearch_WiF():
    print('###########DO GridSearch for WiFF################')
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)


    # default # subsample size: 256; Tree height：8; Number of trees: 100
    num_subsample = [16, 32, 64, 128, 256, 512]  # phi
    num_tree = [10, 30, 50, 70, 90, 100, 110]  # t , 110
    anomaly_rate = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # anomaly_rate = []  # anomaly_rate = np.linspace(0., 0.5, num=8, retstep=True)  # alpha,闭区间生成8个等间隔数
    # max_depth  =  np.range(2, 15, 1)


    # grid search start
    best_score = 0
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
                iForest_parameters = [t, phi, phi-1, 'rate', 0.5, alpha]  # [itree_num, subsamples, hlim, mode, s0, alpha]

                # m2o: rf
                mode = [iForest_parameters, 'M2O_CPDP', 'WiF']

                df_r_measures = WiFF_main(mode=mode, clf_index=2, runtimes=30)

                measures.update({'t': t,
                                 'phi': phi,
                                 'alpha': alpha,
                                 'Balance': df_r_measures['Balance'].mean(),
                                 'G-Mean2': df_r_measures['G-Mean2'].mean(),
                                 'G_Measure': df_r_measures['G-Measure'].mean()})
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
                score_gm2 = df_r_measures['G-Mean2'].mean()
                score_gm = df_r_measures['G-Measure'].mean()


                if score_bal > best_score_bal:
                    best_score_bal = score_bal
                    best_para_bal = {'t': t, "phi": phi, "alpha": alpha}
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

    print('Best socre_gm2:{:.2f}'.format(best_score_gm2))
    print('Best para_gm2:{}'.format(best_para_gm2))

    print('Best socre_gm:{:.2f}'.format(best_score_gm))
    print('Best para_gm:{}'.format(best_para_gm))

    # print("measure2:", df_para_measures)  # 所有不同参数组合的所有结果
    pathname = spath + '\\' + 'para_results.csv'
    df_para_measures.to_csv(pathname)  # save all results of different parameter combinations





if __name__ == '__main__':
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    starttime = time.clock()  # 计时开始

    GridSearch_WiF()

    endtime = time.clock()  # 计时开始
    totaltime = endtime - starttime
    print('程序结束运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序共运行时间为:', totaltime)


