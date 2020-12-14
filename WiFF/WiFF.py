#!/usr/bin/env python
# -*- coding: UTF-8 -*-


"""
=====================================================================
Weight iForest
=====================================================================
The code is about Weighted iForest and iForest.

@author: husky88 and Can Cui
@link: https://blog.csdn.net/husky66/article/details/91872283###
@Webset: CSDN
@E-mail: cuican1414@buaa.edu.cn/cuican5100923@163.com/cuic666@gmail.com
=====================================================================
"""

# print(__doc__)


import math
import random
import copy
import time
import os
from collections import Counter
import pandas as pd
import heapq
import numpy as np

from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model

from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Standard_Features # Process_Data,


class Input_Datas(object):
    """
    该方法用于导入数据，与选取子样本
    """
    def __init__(self, subsample_size):
        # 子样本大小
        self.subsample_size = subsample_size


    def Input(self, Sample):
        """导入数据，并将数据转换成list格式"""
        # print('----Input true data and array2list...')
        self.Sample = Sample.tolist() #list(Sample)  # original data +synthetic data, Array2List
        self.length = len(Sample)  # the number of (original + synthetic) data
        if self.subsample_size >= self.length:
            self.subsample_size = self.length  # subsample_size <= length
        self.ranges = list(range(len(Sample)))  # the list of the sample index


    def Input2(self, Sample0, Sample):
        '''
        导入数据，并将数据转换成list格式,并找到两数组不同数据的行索引
        :param Sample0: 2D array,
        :param Sample: 2D array, Sample[:len(Sample0)] = Sample0
        :return:
        '''

        # print('----Input data containing true and fake samples and array2list...')
        self.Sample = Sample.tolist()  # list(Sample)  # original data +synthetic data, Array2List
        self.length = len(Sample)  # the number of (original + synthetic) data
        if self.subsample_size >= self.length:
            self.subsample_size = self.length  # subsample_size <= length
        self.ranges = list(range(len(Sample)))  # the list of the sample index

        self.Sample0 = Sample0.tolist()  # list(Sample0)  # original data, Array2List
        self.original_length = len(Sample0)  # the number of original data
        self.original_ranges = list(range(len(Sample0)))   # the list of the sample0 index
        # 根据SMOTE的结果发现，人工合成样本在真实样本之后,注意，一旦样本乱序，下行代码不可用
        self.different_ranges = list(range(len(Sample0), len(Sample)))
        # 样本去重后下列内容可用
        # self.same_sample_index = []
        # for value_index in self.original_ranges:  # find the index of (original + synthetic) data whose value is same as in the original data
        #     value = self.Sample0[value_index]
        #     if value in self.Sample:
        #         location = self.Sample.index(value)  # list为列表的名字；value为查找的值；p为value在list的位置
        #         self.same_sample_index.append(location)  # the list of the original index in the whole data

        # debugging
        # df0 = pd.DataFrame(Sample0)
        # df = pd.DataFrame(Sample)
        # # df0.to_csv(r'D:\2017-2019\data\csv格式\Relink_csv\Relink\Zxing_original.csv')
        # # df.to_csv(r'D:\2017-2019\data\csv格式\Relink_csv\Relink\Zxing_combine.csv')
        # print('The length of the original samples and the new samples and the different index list are:\n',
        #       len(self.original_ranges), len(self.ranges), self.different_ranges, len(self.different_ranges))

    def Subsample(self):
        """从Input处理后的数据中，选取子样本"""
        # print('----Select subsamples from the Input data...')
        if self.subsample_size >= len(self.ranges):
            self.subsample_size = len(self.ranges)

        self.subsample = random.sample(self.ranges, self.subsample_size)  # random subsampling index from all samples index list, return a list
        for temp in self.subsample:
            self.ranges.remove(temp)  # remove the index of the selected subsamples

        if len(self.ranges) <= (self.subsample_size / 2):
            self.subsample_size = len(self.ranges) + self.subsample_size
            self.subsample.extend(self.ranges)  # 如果剩余样本不足子样本的1/2,则认为不足以构建一颗树，则将剩余样本归为上一棵树中
            self.ranges = []

class Select_Attribute(object):
    """
    挑选数据属性，并随机挑选一个该属性中的最大值与最小值
    """

    def __init__(self, sample):
        self.Sample = sample   # sample including d attributes,2D array

    # def __init__(self, sample, r1, r2):
    #     self.Sample = sample   # sample including d attributes,2D array
    #     self.r1 = random.seed(r1 + 1)
    #     self.r2 = random.seed(r2 + 1)

    def random_attribute(self):
        """随机挑选一个属性"""
        # print('----Select a data metric randomly...')
        length = len(self.Sample[0])  # the number of attributes(=Sample.shape[1])
        ranges = list(range(length))   # the index of attributes
        # self.r1 = self.r1

        self.random_attribute_datas = random.sample(ranges, 1)  # a random attribute index, return a list containing an element

    def random_values(self, Sample):  # Sample: a list, len(Sample) = subsmaple_size
        """在所挑选属性中随机挑选一个值"""
        # print('----Select a split point from [min,max] values randomly...')
        i = 0
        max = self.Sample[Sample[0]][self.random_attribute_datas[0]]
        min = max
        while i < len(Sample):
            if self.Sample[Sample[i]][self.random_attribute_datas[0]] > max:
                max = self.Sample[Sample[i]][self.random_attribute_datas[0]]

            if self.Sample[Sample[i]][self.random_attribute_datas[0]] < min:
                min = self.Sample[Sample[i]][self.random_attribute_datas[0]]
            i += 1
        # random.random():random float number, [0, 1.0)
        # self.r2 = self.r2

        self.attribute_value = max - random.random() * (max - min)   # random selected a split value

class ITree(object):
    """
    建立孤立树
    """
    def __init__(self, depth, subsample, Sample):
        self.root = subsample  # subsample index list
        self.depth = depth  # limit height
        self.Sample = Sample  # sample index list

    def itree(self):
        """建立隔离树"""
        # print('----Build an iTree...')

        attribute = 0  # initialize the index of the attribute
        depth = 0   # initialize the depth of a sample in an iTree
        self.Tree_1 = []  # 初始化最终构建的隔离树，由嵌套列表构建，即[[节点1中包含的所有样本索引列表,节点1到根节点的深度]，[节点2中包含的所有样本索引列表,节点2到根节点的深度]]
        # 第n1个root=self.Tree[0][n1][0], depth =self.Tree[0][n1][1], attribute=self.Tree[0][n1][2], n1表示Tree的某个节点
        self.Tree = [[self.root, 0, attribute]]
        # print('the sample number of the first node', len(self.Tree[0][0]))
        # it = -1

        while self.Tree and (depth <= self.depth):
            self.left = []
            self.right = []
            # it += 1
            # if it < 15:  # check the node of an iTree to undestand the build processing
            #     print('The', it, 'th built itree is: ', self.Tree, '\n', len(self.Tree[0][0]), '\n')

            # root: a subsample index list; depth: a number; attribute: a attribute index
            root, depth, attribute = self.Tree.pop(0)  # pop()函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
            # random.seed(r+1)
            set_attribute = Select_Attribute(self.Sample)
            # set_attribute = Select_Attribute(self.Sample, r1=1, r2=255)  # add random seed to remian the same values of a iTree
            set_attribute.random_attribute()
            attribute = set_attribute.random_attribute_datas[0]  # random selected an attribute (index)
            set_attribute.random_values(root)
            attribute_value = set_attribute.attribute_value  # random selected a split value of the attribute

            i = 0
            while i < len(self.Sample[0]):  # the number of an attribute
                j = 0
                while j < len(self.root):  # root: a subsample index list
                    if self.Sample[self.root[0]][i] == self.Sample[self.root[j]][i]:  # 属性值相同，判决条件为真，构建树结束
                        self.judge = True
                    else:
                        self.judge = False  # 属性值不相同，判决条件为假，跳出循环，继续构建树
                        break
                    j += 1
                if self.judge == False:  # 判决条件为假，跳出循环，继续构建树
                    break
                i += 1

            i = 0
            while i < len(root):
                # only a sample; the current height is larger than the cutoff l; the stop condition is False
                if (len(root) == 1) or (depth == self.depth - 1) or (self.judge):
                    self.Tree_1.append([root, depth + 1])  # a nested list
                    # print('depth and root and judge', depth, root, self.judge)
                    # print('Reach one of the stop conditions')
                    break
                # a split value (p): attribute_value; an attribute value (q): Sample[][]
                if self.Sample[root[i]][attribute] < attribute_value:  # root[i]: the index of the sample
                    self.left.append(root[i])
                else:
                    self.right.append(root[i])
                i += 1
            depth += 1

            if not (self.left == []):
                self.Tree.append([self.left, depth, attribute])
            if not (self.right == []):
                self.Tree.append([self.right, depth, attribute])

        # print('the built itrees_1 are: ', self.Tree_1, len(self.Tree_1), '\n')


    def weight_itree(self, original_ranges, different_ranges, r):
        '''
        建立权重隔离树
        :param Sample0_index:
        :param Sample_index:
        :return:
        '''
        # print('----Build a weight iTree...')
        attribute = 0  # initialize the index of the attribute
        depth = 0  # initialize the depth of a sample in an iTree
        self.Tree_1 = []  # 初始化最终构建的隔离树，由嵌套列表构建，即[[节点1中包含的所有样本索引列表,节点1到根节点的深度]，[节点2中包含的所有样本索引列表,节点2到根节点的深度]]
        # root=self.Tree[0][n1][0], depth =self.Tree[0][n1][1], attribute=self.Tree[0][n1][2], n1表示Tree的某个节点
        self.Tree = [[self.root, 0, attribute]]
        # print('the sample number of the first node', len(self.Tree[0][0]))
        # it = -1

        while self.Tree and (depth <= self.depth):
            self.left = []
            self.right = []
            # it += 1
            # if it < 15:  # check the each node of an iTree
            #     print('The', it, 'th built itree is: ', self.Tree, '\n', len(self.Tree[0][0]), '\n')

            # root: a subsample index list; depth: a number; attribute: a attribute index
            root, depth, attribute = self.Tree.pop(0)
            # random.seed(r+1)
            set_attribute = Select_Attribute(self.Sample)
            set_attribute.random_attribute()
            attribute = set_attribute.random_attribute_datas[0]  # random selected an attribute (index)
            set_attribute.random_values(root)
            attribute_value = set_attribute.attribute_value  # random selected a split value of the attribute
            r += 1
            true_sample_number = 0  # the
            fake_sample_number = 0
            for rt in range(len(root)):
                one_sample_index = root[rt]  # root = self.Tree[0][0]
                if one_sample_index in original_ranges:  # true sample
                    true_sample_number += 1
                elif one_sample_index in different_ranges:  # synthetic sample
                    fake_sample_number += 1
                else:
                    print('IndexError: Please check the sample index in Sample0_index and Sample_index...')
                    break
            weight = true_sample_number / (true_sample_number + fake_sample_number)  #

            '''the stop conditions:(1)only a sample; 
            (2)the current height is larger than the cutoff l;
            (3) the attribue values of all samples are same;'''
            # the stop conditions(3):self.judge;;
            i = 0
            while i < len(self.Sample[0]):  # the number of an attribute
                j = 0
                while j < len(self.root):  # root: a subsample index list
                    if self.Sample[self.root[0]][i] == self.Sample[self.root[j]][i]:  # 属性值相同，判决条件为真，构建树结束
                        self.judge = True
                    else:
                        self.judge = False  # 属性值不相同，判决条件为假，跳出循环，继续构建树
                        break
                    j += 1
                if self.judge == False:  # 判决条件为假，跳出循环，继续构建树
                    break
                i += 1

            i = 0
            while i < len(root):
                # the stop conditions:(1)(2)(3)
                if (len(root) == 1) or (depth == self.depth - weight) or (self.judge):
                    self.Tree_1.append([root, depth + weight])  # a nested list
                    # print('Reach one of the stop conditions')
                    break
                # a split value (p): attribute_value; an attribute value (q): Sample[][]
                if self.Sample[root[i]][attribute] < attribute_value:  # root[i]: the index of the sample
                    self.left.append(root[i])
                else:
                    self.right.append(root[i])
                i += 1
            depth = depth + weight

            if not (self.left == []):
                self.Tree.append([self.left, depth, attribute])
            if not (self.right == []):
                self.Tree.append([self.right, depth, attribute])

        # print('the built itrees_1 are: ', self.Tree_1, len(self.Tree_1), '\n')


    def prediction(self):
        """计算每一个数据的路径长度
        E(h(x))表示实例x在多个树中的树高h(x)的平均值，
        s(x, n) = 2**−E(h(x))/c(n), c(n) = 2H(n − 1) − (2(n − 1)/n)
        where H(i) is the harmonic number and it can be estimated by ln(i) + 0.5772156649 (Euler’s constant).
        As c(n) is the average of h(x) given n, we use it to normalise h(x).
        • when E(h(x)) ->c(n), s -> 0.5;
        • when E(h(x)) -> 0, s -> 1;
        • and when E(h(x)) -> n − 1, s -> 0."""
        # H(n-1) = ln(n-1) + 0.5772156649， 通过log(x[, base])来设置底数
        # print('self.root:', self.root)
        self.cn = 2 * (math.log(len(self.root) - 1, math.e) + 0.5772156649) \
                  - (2 * (len(self.root) - 1) / (len(self.root)))

        # print('Calculate the depth of each sample in an iTree')
        self.original = sorted(self.root)  # 对子样本索引从小到大进行排序，返回一个列表
        self.path = []
        i = 0
        while i < len(self.original):
            self.path.append(0)
            i += 1

        i = 0
        while i < len(self.original):  # 子样本的数量
            j = 0
            while j < len(self.Tree_1):  # iTree的节点数量，Tree_1[[[root1], depth1 + 1],[[root2], depth2 + 1],...]
                k = 0
                while k < len(self.Tree_1[j][0]):  # 每个节点上的样本数量
                    # print(self.Tree_1, type(self.Tree_1))
                    if self.Tree_1[j][0][k] == self.original[i]:  # 如果第j个节点上的第k个样本索引=子样本索引排序中的序号
                        self.path[i] = self.Tree_1[j][1]  # 将第j个节点上的第k个样本的深度赋值给路径长度path
                    else:
                        pass
                    k += 1
                j += 1
            i += 1

class IForest(object):
    """
    建立多棵树，并求出每一个数据的异常分数
    """

    def __init__(self, Number, subsample_size, max_depth):

        self.number = Number  # 孤立树数量
        self.subsample_size = subsample_size  # 最大子样本
        self.max_depth = max_depth  # 树最大高度


    def Build_Forest(self, Sample, mode, r, s0, alpha):
        '''
        :param Sample:
        :param mode:
        :param s0: 异常得分的阈值
        :param alpha:
        :return:
        '''
        # print('----Build an iForest...')
        random.seed(r+1)
        ranges = [0]  # ranges = []
        self.scores_1 = []  # score list about all samples according to the iForest instead of the sample order
        self.index = []  # sample index about all iTrees
        example_a = Input_Datas(self.subsample_size)
        example_a.Input(Sample)
        data_length = example_a.length

        # limit the itree number according to the sample number
        tree_number = math.ceil(data_length / self.subsample_size)  # without replacement sampling
        if self.number >= tree_number:
            self.number = tree_number
        # print('the number of iTrees is %d' % (self.number))

        if self.max_depth >= self.subsample_size:
            self.max_depth = self.subsample_size-1  # the value in the original paper
        # print('the limit height of iTrees is %d' % (self.max_depth))


        m = 0
        while ranges:
            example_a.Subsample()  # randomly select subsamples
            ranges = example_a.ranges  # remove the index of the selected subsamples
            example_b = ITree(self.max_depth, example_a.subsample, example_a.Sample)
            m += 1
            # print('m', m, ranges)
            level_path = []  # average path length list
            score = []  # abnormal score list
            # initialize the average path list and the abnormal score list
            j = 0
            while j < example_a.subsample_size:
                level_path.append(0)  # initialize average path (make sure the length of the level_path list )
                score.append(0)   # initialize abnormal score (make sure the length of the score list )
                j += 1

            # calculate the whole path length of each sample in the all iTrees
            # i = 0
            # while i < self.number:
            #
            #     i += 1

            example_b.itree()
            example_b.prediction()
            k = 0
            while k < example_a.subsample_size:
                level_path[k] = level_path[k] + example_b.path[
                    k]  # calculate the path length of each sample from subsamples
                k += 1

            k = 0
            while k < example_a.subsample_size:
                level_path[k] = level_path[k] / self.number  # the average length of a sample in the all iTrees; E(h(x))表示实例x在多个树中的树高h(x)的平均值
                score[k] = 2 ** (-level_path[k] / example_b.cn)  # the abnormal score of every sample
                k += 1

            # transfer local variables to global variables
            for temp in score:
                self.scores_1.append(temp)  # all abnormal scores of subsamples
            for temp in example_b.original:
                self.index.append(temp)  # Rank the index of the sample from small to large


        score_in_order, score_in_itree, index_in_itree = self.sort_1(self.scores_1, self.index)  # s is the score results according to the original index

        if mode == 'cutoff':
            abnormal_rate, abnormal_list = self.abnormal_rate(data_length, score_in_order,
                                                     s0)  # when s>s0, calculate abonormal rate
            return score_in_order, abnormal_rate, abnormal_list
        elif mode == 'rate':
            abnormal_rate, abnormal_list = self.contamination_level(data_length, score_in_order,
                                                           alpha)  # when contamination_level=alpha, calculate the contamination_level.
            return score_in_order, abnormal_rate, abnormal_list
        else:
            print('mode is wrong, please check...')
            return None

    def Build_Weight_Forest(self, Sample0, Sample, mode, r, s0, alpha):  #  r1, r2
        '''
        建立权重孤立森林

        :param Sample0:
        :param Sample:
        :param mode:
        :param r:
        :param s0: 异常得分的阈值
        :param alpha:
        :return:
        '''

        # print('----Build a weight iForest...')
        random.seed(r+1)
        ranges = [0]
        original_ranges = [0]
        self.scores_1 = []  # score list about all samples according to the iForest instead of the sample order
        self.index = []  # sample index about all iTrees
        example_a = Input_Datas(self.subsample_size)  # the subsample size
        example_a.Input2(Sample0, Sample)  # input Sample0 and Sample
        data_length = example_a.length  # the number of true + fake samples
        data_original_length = example_a.original_length  # the number of true samples

        # limit the itree number according to the sample number
        tree_number = math.ceil(data_length / self.subsample_size)  # without replacement sampling
        if self.number >= tree_number:
            self.number = tree_number
        # print('the number of iTrees is %d' % (self.number))

        if self.max_depth >= self.subsample_size:
            self.max_depth = self.subsample_size - 1  # the value in the original paper
        # print('the limit height of iTrees is %d' % (self.max_depth))

        while ranges and original_ranges:
            example_a.Subsample()  # randomly select subsamples
            original_ranges = example_a.original_ranges  # the list of the same index in both Sample0 and Sample
            different_ranges = example_a.different_ranges  # the list of the index in Sample and not in Sample0
            ranges = example_a.ranges  # Sample_index, i.e., remove the index of the selected subsamples
            # print(m, original_ranges, len(original_ranges), ranges, len(ranges))  # len(ranges)=143,
            example_b = ITree(self.max_depth, example_a.subsample, example_a.Sample)  # build iTree
            # r += 1 # r1 += 1
            level_path = []  # average path length list
            score = []  # abnormal score list
            # initialize the average path list and the abnormal score list
            j = 0
            while j < example_a.subsample_size:
                level_path.append(0)  # initialize average path (make sure the length of the level_path list )
                score.append(0)  # initialize abnormal score (make sure the length of the score list )
                j += 1

            # calculate the whole path length of each sample in the all iTrees
            # i = 0
            # while i < self.number:
            #
            #     i += 1
            example_b.weight_itree(original_ranges, different_ranges, r)
            example_b.prediction()
            k = 0
            while k < example_a.subsample_size:
                level_path[k] = level_path[k] + example_b.path[
                    k]  # calculate the path length of each sample from subsamples
                k += 1

            # calculate the abnormal score of each sample in the all iTrees
            k = 0
            while k < example_a.subsample_size:
                level_path[k] = level_path[
                                    k] / self.number  # the average length of a sample in the all iTrees; E(h(x))表示实例x在多个树中的树高h(x)的平均值
                score[k] = round(2 ** (-level_path[k] / example_b.cn), 4)  # the abnormal score of every sample
                k += 1

            for temp in score:
                self.scores_1.append(temp)  # all abnormal scores of subsamples
            for temp in example_b.original:
                self.index.append(temp)  # Rank the index of the sample from small to large
            # print("rank:", m, example_b.original, len(example_b.original), '\n')
            # m += 1
        score_in_order, score_in_itree, index_in_itree = self.sort_1(self.scores_1,
                                                                     self.index)  # s is the score results according to the original index
        if mode == 'cutoff':
            abnormal_rate, abnormal_list = self.abnormal_rate(data_length, score_in_order, s0)  # when s>s0, calculate abonormal rate
            return score_in_order, abnormal_list
        elif mode == 'rate':
            abnormal_rate, abnormal_list = self.contamination_level(data_length, score_in_order,
                                                 alpha)  # when contamination_level=alpha, calculate the contamination_level.
            return score_in_order, abnormal_rate, abnormal_list
        else:
            print('mode is wrong, please check...')
            return None
        # print('样本的异常得分0为', score_in_order)



    def sort_1(self, scores_1, index):
        """排序"""
        # print('----sort the abnormal scores of all samples...')
        score = [0] * len(index)  # initialize the all  abnormal scores list contianing len(subsample) elements
        # print('scores_1:', scores_1)  # scores_1 is not in order
        for i in range(len(index)):
            # 将对应样本的得分填入对应索引的位置
            true_index = index[i]
            score[true_index] = round(scores_1[i], 4)  # save the score in the corresponding location according to the index

        return score, scores_1, index

    def abnormal_rate(self, data_length, score_in_order, s0):
        '''
        calculate the abnormal rate when the threshold of an outlier is s0
        :param s: type: list, the abnormal score about the input data according to the sample index.
        :param sample_number: the number of the input data.
        :return: alpha: abnormal rate whose type is float, the value range is [0.0, 1.0].
                 abnormal_list: type: list, abnormal sample index
        '''
        if len(score_in_order) == data_length:
            abnormal_number = 0
            abnormal_list = 0
            for i in range(len(score_in_order)):
                if score_in_order[i] >= s0:
                    abnormal_number += 1
                    abnormal_list.append(score_in_order[i])
            alpha = (abnormal_number / len(score_in_order))
            # print("The number of samples whose score are larger than %.4f is %d;\n" % (s0, sum),
            #       "The abnormal rate is %.4f" % rate)
            abnormal_list.sort()
            return alpha, abnormal_list
        else:
            print('Please check the samples number and the subsamples number')
            return None

    def contamination_level(self, data_length, score_in_order, alpha):
        # 向上取整 # math.floor # 向下取整 # round(f,n)  # 四舍五入,n位小数
        contamination_number = math.ceil(data_length * alpha)
        # scores = copy.deepcopy(score_in_order)  # deep copy
        scores = copy.copy(score_in_order)  # shallow copy
        abnormal_list = []
        if alpha != 0:  # exist outliers in the input data
            Inf = 0
            for i in range(contamination_number):
                max_value = max(scores)
                max_index = scores.index(max_value)
                abnormal_list.append(max_index)
                scores[max_index] = Inf
            abnormal_list.sort()
            # print('when the contamination level is %s, the abnormal list is %s' % (alpha, temp))

            # method 2: if there are same values in the list, only return the first index
            # for i in range(contamination_number):
            #     result.sort()
            # result = map(nums.index, heapq.nlargest(contamination_number, nums))  # list中最大几个数对应的索引
            # print('when the contamination level is %s, the abnormal list is %s' % (alpha, result))
            # return result
        else:
            pass
            # return list()
        return alpha, abnormal_list

def WiFF_main(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)
    # example datasets for test
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    # datasets for RQ2
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.4.csv', 'xerces-init.csv'],
    #         ['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv'],
    #         ['EQ.csv', 'JDT.csv', 'LC.csv', 'ML.csv', 'PDE.csv'],
    #         ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    # datasets for parameters discussion
    datasets = [['arc-1.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv']]

    datanum = 0
    for i in range(len(datasets)):
        datanum = datanum + len(datasets[i])
    # print(datanum)
    # mode = [preprocess_mode, train_mode, save_file_name]
    # preprocess_mode = mode[0]
    iForest_parameters = mode[0]
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
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug', r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    data_train = pd.DataFrame(Sample_tr)
                    data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                    data_train_unique = data_train_unique.values
                    X_train = data_train_unique[:, : -1]
                    y_train = data_train_unique[:, -1]
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation
                    target = np.c_[X_test_zscore, y_test]

                    # /* step2: WiFF(weighted iForest Filter) */
                    # *******************WiFF*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train_neg0, X_train_pos0, y_train_neg0, y_train_pos0 = DevideData2TwoClasses(X_train_zscore, y_train)  # original X+,y+,X-,y-
                    X_train_resampled, y_train_resampled = DataImbalance(X_train_zscore, y_train, r, mode='smote')  # Data imbalance
                    X_train_neg, X_train_pos, y_train_neg, y_train_pos = DevideData2TwoClasses(X_train_resampled, y_train_resampled)  # DB X+,y+,X-,y-

                    # iForest_parameters = [itree_num, subsamples, hlim, mode, s0, alpha]
                    itree_num = iForest_parameters[0]
                    subsamples = iForest_parameters[1]
                    hlim = iForest_parameters[2]
                    mode_if = iForest_parameters[3]
                    s0 = iForest_parameters[4]
                    alpha = iForest_parameters[5]

                    forest = IForest(itree_num, subsamples, hlim)  # build an iForest(t, subsample, hlim)
                    s, rate, abnormal_list = forest.Build_Forest(X_train_neg, mode_if, r, s0, alpha)
                    ranges_X_train_neg, Sample_X_train_neg = Delete_abnormal_samples(X_train_neg, abnormal_list)
                    ranges_y_train_neg, Sample_y_train_neg = Delete_abnormal_samples(y_train_neg, abnormal_list)

                    s_weight, rate_weight, abnormal_list_weight = forest.Build_Weight_Forest(X_train_pos0, X_train_pos, mode_if, r, s0, alpha)
                    ranges_X_train_pos_weight, Sample_X_train_pos_weight = Delete_abnormal_samples(X_train_pos, abnormal_list_weight)
                    ranges_y_train_pos_weight, Sample_y_train_pos_weight = Delete_abnormal_samples(y_train_pos, abnormal_list_weight)
                    X_train_new = np.concatenate((Sample_X_train_neg, Sample_X_train_pos_weight), axis=0)  # array垂直拼接
                    y_train_new = np.array(Sample_y_train_neg.tolist() + Sample_y_train_pos_weight.tolist())  # list垂直拼接

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requiresSelection_Classifications the label must beong to {0, 1}.
                    modelname_wiff, model_wiff = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_wiff)
                    # print("modelname_wiff:", modelname_wiff)
                    measures_wiff = Build_Evaluation_Classification_Model(model_wiff, X_train_new, y_train_new, X_test_zscore, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_wiff.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_wiff,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_wiff, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False, ignore_index=False)  # the measures of all files in runtimes

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures


if __name__ == '__main__':

    iForest_parameters = [100, 256, 255, 'rate', 0.5, 0.1]  # default values of iForest

    # mode = [[1, 1, 1, '', 'zscore', 'smote'], 'M2O_CPDP', 'eg_WiFF', iForest_parameters]
    mode = [iForest_parameters, 'M2O_CPDP', 'eg_WiFF']
    foldername = mode[1] + '_' + mode[2]
    # only NB classifier needs to log normalization for WiFF
    WiFF_main(mode=mode, clf_index=0, runtimes=2)

