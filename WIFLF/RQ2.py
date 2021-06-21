#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
Answering RQ2 for Paper: Is WIFLF better than other CPDP-CM methods?

    1)BF: Burak filter (NN-filter), 2008
    2)BNaive: Naive Bellwether, 2019
    3)BTCAplus: Bellwether + TCA+, 2019
    4)BTNB: Bellwether + TNB, 2019
    5)DFAC: ,2015
    6)HISNN: 2015
    7)HSBF:
    8)TCAplus:2013
    9)TNB:Transfer Naive Bayes, 2012
`Weighted Isolation Forest with Label Information Filter for Cross-Project Defect Prediction with Common Metrics`
============================================================================
@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
@2021/4/9
============================================================================
"""
# print(__doc__)
from __future__ import print_function
import pandas as pd
import numpy as np
import math
import heapq
import random
import copy
import time
import os

from BurakFilter import BF_main
from TNB import TNB_main
from TCAplus import TCAplus_main
from HISNN import HISNN_main
from DFAC import DFAC_main
from HSBF import HSBF_main
from Bellwether import Bellwether_main


import multiprocessing
from functools import partial


def main_BF(runtimes):

    mode = ['', 'M2O_CPDP', 'BF']

    # only NB classifier needs to log normalization for BF
    # BF_main(mode=mode, clf_index=0, runtimes=2)
    for i in range(1, 3, 1):
        BF_main(mode=mode, clf_index=i, runtimes=runtimes)

def main_TNB(runtimes):
    mode = ['', 'M2O_CPDP', 'TNB']
    # NB, LR and RF will gain the same results, because TNB just uses NB as the basic learner.
    TNB_main(mode=mode, clf_index=0, runtimes=runtimes)
        

def main_TCAplus(runtimes):
    mode = ['null', 'M2O_CPDP', 'TCAplus']
    for i in range(0, 3, 1):
        TCAplus_main(mode=mode, clf_index=i, runtimes=runtimes)
        
        
def main_HISNN(runtimes):
    mode = ['null', 'M2O_CPDP', 'HISNN']
    for i in range(1, 3, 1):
        HISNN_main(mode=mode, clf_index=i, runtimes=runtimes)

    # NB classifier needs to log normalization for HISNN
    # HISNN_main(mode=mode, clf_index=0, runtimes=2)


def main_DFAC(runtimes):
    mode = ['', 'M2O_CPDP', 'DFAC']
    for i in range(1, 3, 1):
        DFAC_main(mode=mode, clf_index=i, runtimes=runtimes)

def main_HSBF(runtimes):

    mode = ['', 'M2O_CPDP', 'HSBF']
    for i in range(1, 3, 1):
        HSBF_main(mode=mode, clf_index=i, runtimes=runtimes)


def main_NaiveBellwether(runtimes):

    mode = ['null', 'M2O_CPDP', 'BNaive']
    learnername = 'Naive'
    for i in range(0, 3, 1):
        Bellwether_main(learnername=learnername, mode=mode, clf_index=i, runtimes=runtimes)

def main_BellwetherTCAplus(runtimes):

    mode = ['null', 'M2O_CPDP', 'BTCAplus']
    learnername = 'TCA+'
    for i in range(0, 3, 1):
        Bellwether_main(learnername=learnername, mode=mode, clf_index=i, runtimes=runtimes)

def main_BellwetherTNB(runtimes):

    mode = ['null', 'M2O_CPDP', 'BTNB']
    learnername = 'TNB'
    for i in range(0, 3, 1):
        Bellwether_main(learnername=learnername, mode=mode, clf_index=i, runtimes=runtimes)

def somefunc( iterable_term):

    runtimes = 30
    if iterable_term == 1:
        return main_BF(runtimes)
    elif iterable_term == 2:
        return main_TNB(runtimes)
    elif iterable_term == 3:
        a = main_NaiveBellwether(runtimes)
        # return main_BellwetherTCAplus(runtimes)
        return main_TCAplus(runtimes)
    elif iterable_term == 4:
        return main_HISNN(runtimes)
    elif iterable_term == 5:
        return main_DFAC(runtimes)
    elif iterable_term == 6:
        return main_HSBF(runtimes)
    elif iterable_term == 7:
        return main_BellwetherTNB(runtimes)




if __name__ == '__main__':
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    starttime = time.clock()  # 计时开始
    iterable = [1, 2, 3, 4, 5, 6, 7, 8]

    pool = multiprocessing.Pool()
    cores = multiprocessing.cpu_count()
    # print('cores:', cores)
    func = partial(somefunc)
    pool.map(func, iterable)
    pool.close()
    pool.join()

    endtime = time.clock()  # 计时结束

    totaltime = endtime - starttime
    print('程序结束运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('程序共运行时间为:', totaltime)










