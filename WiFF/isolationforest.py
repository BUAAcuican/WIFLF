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

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf


class IsolationTreeEnsemble:  #
    def __init__(self, sample_size, n_trees=10):

        self.sample_size = sample_size            # subsample size
        self.n_trees = n_trees                    # number of trees
        self.height_limit = np.log2(sample_size)  # height limit of an itree
        self.trees = []                           # construct multiple itrees, i.e., iForest

    def fit(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        param: X-        training samples, type-DataFrame
        return iForest
        """
        if isinstance(X, pd.DataFrame):  # isinstance(object, classinfo) to judge whether the type of object is classinfo.
            X, y = X.iloc[:, :-1].values,  X.iloc[:, -1].values  # DataFrame2Array
            len_x = len(X)  # number of instances, X.shape[0]
            col_x = X.shape[1]  # number of features
            self.trees = []  # initialize the iForest

        sample_idx_list = list(range(len(X)))

        # adjust the number of witrees
        if int(len(X) /self.sample_size + 1) > self.n_trees:
            self.n_trees = int(len(X) /self.sample_size + 1)

        if sample_idx_list != []:
            for i in range(self.n_trees):
                if len(sample_idx_list) < self.sample_size:
                    self.sample_size = len(sample_idx_list)

                sample_idx = random.sample(sample_idx_list, self.sample_size)   # randomly select the sub-samples

                if sample_idx_list != []:
                    temp_tree = IsolationTree(self.height_limit, 0).fit(X[sample_idx, :], sample_idx)
                    self.trees.append(temp_tree)

                for i in sample_idx:
                    sample_idx_list.remove(i)

            return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        pl_vector = []                    # initialize path lengths of X
        if isinstance(X, pd.DataFrame):   # type of X is pd.DataFrame
            # X, y = X.iloc[:, :-1], X.iloc[:, -1]
            X = X.values  # DataFrame2Array
            # y = y.values
        for x in (X):

            pl = np.array([path_length_itree(x, t, 0) for t in self.trees])  # x: an instance in X, t: an iTree in iForest, e=0 initialize the current length
            # print(pl)
            pl = pl.mean()

            pl_vector.append(pl)

        pl_vector = np.array(pl_vector).reshape(-1, 1)

        return pl_vector

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        return 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """

        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions

    def predict_threshold(self, X:np.ndarray, threshold: float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."

        scores = 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions

    def predict_percentile(self, X: np.ndarray, percentile: float) -> np.ndarray:
        "Calling anomaly_score() and predict anomaly points by percentile."

        scores = 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))


        topk = int(len(X) * percentile)  # the number of the max value index

        maxscores_list, max_num_index = selectTopkNumbers(scores, topk)

        predictions = [1 if p in max_num_index else 0 for p in range(len(scores))]

        return predictions

class IsolationTree:

    def __init__(self, height_limit, current_height):
        self.height_limit = height_limit       # height limit of weighted-iTree
        self.current_height = current_height   # current height of an instance x
        self.split_by = None                   # split feature index
        self.split_value = None                # split value of the feature
        self.right = None                      # right leaf
        self.left = None                       # left leaf
        self.size = 0                          # number of instance in a weighted-iTree
        self.exnodes = 0                       # external nodes, initialize to 0
        self.n_nodes = 1                       # number of nodes from this layer to the bottom layer, initialize to 1
        self.left_idx = None                   # index of samples on the left node
        self.right_idx = None                  # index of samples on the right node

    def fit(self, X: np.ndarray, X_idx):
        """
        Add Extra while loop
        """

        if len(X) <= 1 or self.current_height >= self.height_limit:
            self.exnodes = 1
            self.size = len(X)
            return self

        split_by = random.choice(np.arange(X.shape[1]))
        min_x = X[:, split_by].min()
        max_x = X[:, split_by].max()

        if min_x == max_x:
            self.exnodes = 1
            self.size = len(X)
            return self

        # split_value = min_x + random.betavariate(0.5, 0.5) * (max_x - min_x)
        split_value = min_x + random.random() * (max_x - min_x)
        a = X[X[:, split_by] < split_value]
        b = X[X[:, split_by] >= split_value]
        tf = np.where(X[:, split_by] < split_value, True, False)  # np.where(condition, x, y), condition->x, not condition->y
        # del X[:, split_by]  # delete the selected feature column
        temp_left_idx = [i for i, tfi in enumerate(list(tf)) if tfi]  # list(tf).index(True)
        temp_right_idx = [i for i, tfi in enumerate(list(tf)) if ~tfi]  # list(tf).index(False)

        self.size = X.shape[0]
        self.split_by = split_by
        self.split_value = split_value

        self.left_idx = []
        for j in temp_left_idx:
            self.left_idx.append(X_idx[j])

        self.right_idx = []
        for j in temp_right_idx:
            self.right_idx.append(X_idx[j])

        self.left = IsolationTree(self.height_limit, self.current_height + 1).fit(a, self.left_idx)
        self.right = IsolationTree(self.height_limit, self.current_height + 1).fit(b, self.right_idx)
        self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self

def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0
    # if n == 0:   # why n==0 occurs?
    #     return 0

def path_length_itree(x, t, e):

    e = e
    if t.exnodes == 1:
        e = e + c(t.size)
        return e
    else:
        a = t.split_by
        if x[a] < t.split_value:
            # n_l = 0
            # for i in x_synthetic_idx:
            #     if i in t.left_idx:
            #         n_l += 1
            # w_l = (len(t.left_idx) - n_l)/len(t.left_idx)
            return path_length_itree(x, t.left, e + 1)

        if x[a] >= t.split_value:
            # n_r = 0
            # for i in x_synthetic_idx:
            #     if i in t.right_idx:
            #         n_r += 1
            # w_r = (len(t.right_idx) - n_r)/len(t.right_idx)
            return path_length_itree(x, t.right, e + 1)
def selectTopkNumbers(scores, topk):
    '''

    param: scores
    param: topk
    '''

    tmp_scores_list = copy.deepcopy(list(scores))  # deepcopy list: scores to avoid changing scores
    sorted_scores_list = []
    max_num_index = []
    for i in range(len(tmp_scores_list)):
        index_i = tmp_scores_list.index(max(tmp_scores_list))  # get the max value and the index of 'scores'
        max_num_index.append(index_i)  # append the index in 'max_num_index'
        tmp_scores_list[index_i] = float('-inf')  # set "-inf" when scanning the max value of scores to avoid selecting it again
        sorted_scores_list.append(scores[max_num_index[i]])
    # print("the index from the max to the min:", max_num_index)
    # print("the values from the max to the min:", sorted_scores_list)

    selected_scores_list = []
    # print("topk:", topk)

    if topk < len(sorted_scores_list):
        temp_value = sorted_scores_list[topk-1]     # the last max value
        if temp_value == sorted_scores_list[topk]:  # the last max has many equal values
            temp_idex = []  # temp index of sorted_scores_list
            n = sorted_scores_list.index(temp_value)  # the first location of the last max value
            for s in range(len(sorted_scores_list)):
                if sorted_scores_list[s] == temp_value:
                    temp_idex.append(max_num_index[s])

            # print(temp_idex, n)   # = selected_index+
            selected_index = max_num_index[:n] + random.sample(temp_idex, topk-n)
            for l in selected_index:
                selected_scores_list.append(scores[l])
        else:
            selected_scores_list = sorted_scores_list[: topk]
            selected_index = max_num_index[: topk]
    else:
        selected_scores_list = sorted_scores_list
        selected_index = max_num_index
    return selected_scores_list, selected_index
    # print("scores:", selected_scores_list, '\n', "original_scores_idx:", selected_index)

def detect_anomalies(X:pd.DataFrame, sample_size=256, n_trees = 100, threshold=0.5, percentile=0.1):

    N = len(X)
    it = IsolationTreeEnsemble(sample_size=sample_size, n_trees=n_trees)
    fit_start = time.time()
    it.fit(X)    # note: input is original samples: X
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    # print(f"fit time {fit_time:3.2f}s")

    if isinstance(X, pd.DataFrame):
        X_real, y_real = X.iloc[:, :-1].values, X.iloc[:, -1].values
        # print("X_real, y_real:", len(X_real),  len(y_real), '\n', y_real)
        pathlength_start = time.time()
        pathlengths = it.path_length(X_real)
        # print("path lengths:\n", pathlengths, len(pathlengths))
        pathlength_stop = time.time()
        pathlength_time = pathlength_stop - pathlength_start
        # print(f"path length time {pathlength_time:3.2f}s")

        score_start = time.time()
        scores = it.anomaly_score(X_real)
        # print("anomaly scores:\n", scores, len(scores))
        score_stop = time.time()
        score_time = score_stop - score_start
        # print(f"score time {score_time:3.2f}s")

        y_predict1 = it.predict_threshold(X_real, threshold)     # note: input is smoted samples: X_smote
        y_predict2 = it.predict_percentile(X_real, percentile)   # note: input is smoted samples: X_smote
        # print(y_predict1, '\n', y_predict2)

        leave_X, leave_y = [], []
        left_idx = []
        for i in range(len(y_predict2)):
            if y_predict2[i] == 0:
                leave_X.append(X_real[i])
                leave_y.append(y_real[i])
                left_idx.append(i)

        leave_X = np.array(leave_X)
        leave_y = np.array(leave_y)
        # leftData = np.c_[leave_X, leave_y]
        # print(left_idx, '\n',  leave_y, '\n', leftData)

        return left_idx, leave_X, leave_y




if __name__ == '__main__':

    np.random.seed(21)
    random.seed(21)
    r = 5
    # print(c(0), c(1), c(2), c(3), '\n',  c(4), c(5), c(9))
    datafile = 'cancer.csv'
    df = pd.read_csv(datafile)
    targetcol = 'diagnosis'
    sample_size = int(16)
    n_trees = int(2)

    allinstancesFlag = True
    subset_size = 32
    if allinstancesFlag:
        N = len(df)
    else:
        N = int(subset_size)

    # df = df.sample(N)  # grab random subset (too slow otherwise)
    df = df.iloc[103:135, :]
    DATA = df
    # print("InputData:", DATA)
    # X, y = df.drop(targetcol, axis=1), df[targetcol]
    left_idx, leftX, lefty = detect_anomalies(DATA, sample_size=sample_size, n_trees=n_trees,threshold=0.65, percentile=0.1)
    print("if:", left_idx, leftX, lefty, len(lefty))