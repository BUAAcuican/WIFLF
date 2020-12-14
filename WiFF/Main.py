#!/usr/bin/env python
# -*- coding: UTF-8 -*-



# from HISNN import HISNN_train, HISNN_test

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
============================================================================
Main:
============================================================================
@author: Can Cui
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
@2019/12/27
============================================================================
"""
# print(__doc__)

import pandas as pd
import numpy as np
import math
import heapq
import random
import copy
import time
import os

from WiFF import WiFF_main
from BurakFilter import BF_main
from HISNN import HISNN_main
from DFAC import DFAC_main
from HSBF import HSBF_main
from Bellwether import Bellwether_main


import multiprocessing
from functools import partial

def Main_WiFF(runtimes):
    # iForest_parameters = [100, 256, 255, 'rate', 0.5, 0.1]  # default values of iForest
    iForest_parameters = [100, 256, 255, 'rate', 0.5, 0.6]  # acceptable values of WiFF

    mode = [iForest_parameters, 'M2O_CPDP', 'eg_WiFF']
    for i in range(1, 3, 1):
        WiFF_main(mode=mode, clf_index=i, runtimes=runtimes)

def MAIN_BF(runtimes):

    mode = ['', 'M2O_CPDP', 'eg_BF' ]

    # only NB classifier needs to log normalization for BF
    # BF_main(mode=mode, clf_index=0, runtimes=2)

    for i in range(1, 3, 1):
        BF_main(mode=mode, clf_index=i, runtimes=runtimes)



def MAIN_HISNN(runtimes):
    mode = ['null', 'M2O_CPDP', 'eg_HISNN']
    for i in range(1, 3, 1):
        DFAC_main(mode=mode, clf_index=i, runtimes=runtimes)

    # NB classifier needs to log normalization for HISNN
    # HISNN_main(mode=mode, clf_index=0, runtimes=2)


def MAIN_DFAC(runtimes):
    mode = ['', 'M2O_CPDP', 'eg_DFAC']
    for i in range(1, 3, 1):
        DFAC_main(mode=mode, clf_index=i, runtimes=runtimes)

def MAIN_HSBF(runtimes):

    mode = ['', 'M2O_CPDP', 'eg_HSBF']
    for i in range(1, 3, 1):
        HSBF_main(mode=mode, clf_index=i, runtimes=runtimes)


def MAIN_Bellwether(runtimes):

    mode = ['null', 'M2O_CPDP', 'eg_Bellwether']
    learnername = 'Naive'
    for i in range(1, 3, 1):
        Bellwether_main(learnername=learnername, mode=mode, clf_index=i, runtimes=runtimes)

def somefunc( iterable_term):
    runtimes = 2
    if iterable_term == 6:
        return MAIN_BF(runtimes)
    elif iterable_term == 2:
        return MAIN_HISNN(runtimes)
    elif iterable_term == 4:
        return MAIN_Bellwether(runtimes)  #
    elif iterable_term == 1:
        return MAIN_HSBF(runtimes)
    elif iterable_term == 3:
        return MAIN_DFAC(runtimes)
    elif iterable_term == 5:
        return Main_WiFF(runtimes)


if __name__ == '__main__':
    print('以下为程序打印所有内容：')
    print('程序开始运行时间', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    starttime = time.clock()  # 计时开始
    iterable = [1, 2, 3, 4, 5, 6]

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










