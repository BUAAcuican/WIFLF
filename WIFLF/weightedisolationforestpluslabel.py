#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
import pandas as pd
import random
import time
import copy
from collections import Counter

from imblearn.over_sampling import SMOTE


class WeightedIsolationTreeEnsemble:  #
    def __init__(self, sample_size, n_trees=10):

        self.sample_size = sample_size            # subsample size
        self.n_trees = n_trees                    # number of trees
        self.height_limit = np.log2(sample_size)  # height limit of an itree
        self.trees = []                           # construct multiple itrees, i.e., iForest

    def fit_improved(self, X_all:np.ndarray, sample_idx_list, sample_synthetic_idx_list):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        param: X_all-                     training samples with different labels to build weighted-iForest, type-np.array
        param: sample_idx_list-           index list of the X
        param: sample_synthetic_idx_list- the index list of synthetic samples by SMOTE
        return iForest
        """

        # adjust the number of witrees
        if int(len(X_all) / (2 * self.sample_size) + 1) > self.n_trees:
            self.n_trees = int(len(X_all) / (2 * self.sample_size) + 1)

        if sample_idx_list != []:
            for i in range(self.n_trees):
                if len(sample_idx_list) < self.sample_size:
                    self.sample_size = len(sample_idx_list)

                sample_idx = random.sample(sample_idx_list, self.sample_size)  # randomly select the sub-samples

                if sample_idx_list != []:
                    temp_tree = WeightedIsolationTree(self.height_limit, 0).fit_improved(X_all[sample_idx, :],
                                                                                     sample_idx,
                                                                                     sample_synthetic_idx_list)
                    self.trees.append(temp_tree)
                for i in sample_idx:
                    sample_idx_list.remove(i)

            return self

    def path_length_improved(self, X:np.ndarray) -> np.ndarray:
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

            pl = np.array([path_length_witree(x, t, 0) for t in self.trees])  # x: an instance in X, t: an iTree in iForest, e=0 initialize the current length
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
        return 2.0 ** (-1.0 * self.path_length_improved(X) / c(len(X)))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """

        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions

    def predict_threshold(self, X:np.ndarray, threshold: float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."

        scores = 2.0 ** (-1.0 * self.path_length_improved(X) / c(len(X)))

        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions

    def predict_percentile(self, X: np.ndarray, percentile: float) -> np.ndarray:
        "Calling anomaly_score() and predict anomaly points by percentile."

        scores = 2.0 ** (-1.0 * self.path_length_improved(X) / c(len(X)))

        topk = int(len(X) * percentile)  # the number of the max value index

        maxscores_list, maxvalue_index = selectTopkNumbers(scores, topk)

        predictions = [1 if p in maxvalue_index else 0 for p in range(len(scores))]

        return predictions

class WeightedIsolationTree:

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
        self.w = 0                             # the weight of each node

    def fit_improved(self, X: np.ndarray, Selected_X_all_idx, sample_synthetic_idx_list):
        """
        param: X-                         randomly selected training samples with single labels to build weighted-iTree, type-np.array.
        param: X_idx-                     index list of the X in X_all.
        param: sample_synthetic_idx_list- the index list of synthetic samples by SMOTE.
        """
        n_sample_synthetic = 0   # number of synthetic samples in the selected X
        for i in Selected_X_all_idx:
            if i in sample_synthetic_idx_list:
                n_sample_synthetic += 1

        # if len(X)== 0:
        #     self.w = 1
        # else:
        #     self.w = (len(X) - n_sample_synthetic) / len(X)

        self.w = (len(X) - n_sample_synthetic) / len(X)

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
            self.left_idx.append(Selected_X_all_idx[j])

        self.right_idx = []
        for j in temp_right_idx:
            self.right_idx.append(Selected_X_all_idx[j])

        self.left = WeightedIsolationTree(self.height_limit, self.current_height + self.w*1).fit_improved(a, self.left_idx, sample_synthetic_idx_list)
        self.right = WeightedIsolationTree(self.height_limit, self.current_height + self.w*1).fit_improved(b, self.right_idx, sample_synthetic_idx_list)
        self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self

def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0

def path_length_witree(x, t, e):

    e = e
    w = t.w
    if t.exnodes == 1:
        e = e + w * c(t.size)
        return e
    else:
        a = t.split_by
        if x[a] < t.split_value:
            # n_l = 0
            # for i in x_synthetic_idx:
            #     if i in t.left_idx:
            #         n_l += 1
            # w_l = (len(t.left_idx) - n_l)/len(t.left_idx)
            return path_length_witree(x, t.left, e + w * 1)

        if x[a] >= t.split_value:
            # n_r = 0
            # for i in x_synthetic_idx:
            #     if i in t.right_idx:
            #         n_r += 1
            # w_r = (len(t.right_idx) - n_r)/len(t.right_idx)
            return path_length_witree(x, t.right, e + w * 1)

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

def detect_anomalies_improvedplusLabel(X:pd.DataFrame,random_state, sample_size=256, n_trees = 100, threshold=0.5, percentile=0.1):
    """
    Function detect_anomalies_improvedplusLabel() summary goes here:
    Remove samples that are justified as abnormal samples and return the normal samples.
    step1: divide samples into positive and negative samples;
    step2: employ wiforest in positive and negative samples without labels, respectively;
    step3: obtain abnormal samples from step2;
    step4: stack negatives and positive samples with labels.
    @param: X-           type: pd.DataFrame, input samples with labels
    @param: random_state-type: int (>0), random state
    @param: sample_size- type: int, subsampling size of WiForest, default: 256
    @param: n_trees-     type: int, number of witrees in WiForest, default: 100
    @param: threshold-   type: float, threshold to justfy the normal and abnormal samples in X, default=0.5
    @param: percentile-  type: float, percentile to justify the abnormal samples in X, default=0.1
    @return: left_idx-   type: list, index of left samples after wiforest
             left_X-     type: 2D array, index of left features of samples after wiforest
             left_y-     type: 1D array, left labels of samples after wiforest
    """


    if isinstance(X, pd.DataFrame):
        X_real, y_real = X.iloc[:, :-1].values, X.iloc[:, -1].values
        # print("X_real, y_real:", len(X_real),  len(y_real), '\n', y_real)
        num = int(len(y_real) - np.sum(y_real))
        # print("num of negative, positive and all samples:", num, int(np.sum(y_real)), len(y_real))

        X_smote, y_smote = SMOTE(random_state=random_state).fit_sample(X_real, y_real)

        # ratio='auto',random_state=0,k_neighbors=2,m_neighbors=10,out_step=0.5,kind='regular',svm_estimator=None,n_jobs=1
        y_smote_class = sorted(Counter(y_smote).items())
        # print("X_smote, y_smote:", y_smote_class)  # , '\n', y_smote)

        sample_idx_list = list(range(len(X_smote)))
        sample_synthetic_idx_list = list(range(len(X), len(X_smote)))

        X_positive = []
        X_negative = []
        X_positive_idx_list0 = []
        X_negative_idx_list0 = []
        for i in range(len(X_smote)):
            if y_smote[i] == 1:
                X_positive.append(X_smote[i])
                X_positive_idx_list0.append(i)
            else:
                X_negative.append(X_smote[i])
                X_negative_idx_list0.append(i)

        X_negative = np.array(X_negative)
        X_positive = np.array(X_positive)
        X_positive_idx_list = copy.deepcopy(X_positive_idx_list0)
        X_negative_idx_list = copy.deepcopy(X_negative_idx_list0)

        N = len(X_negative) * 2
        wpit = WeightedIsolationTreeEnsemble(sample_size=sample_size, n_trees=n_trees)
        wnit = WeightedIsolationTreeEnsemble(sample_size=sample_size, n_trees=n_trees)
        fit_start = time.time()
        wpit.fit_improved(X_smote, X_positive_idx_list0, sample_synthetic_idx_list)
        wnit.fit_improved(X_smote, X_negative_idx_list0, sample_synthetic_idx_list)
        fit_stop = time.time()
        fit_time = fit_stop - fit_start
        print(f"fit time {fit_time:3.2f}s")

        pathlength_start = time.time()
        pathlengths_positive = wpit.path_length_improved(X_positive)
        pathlengths_negative = wnit.path_length_improved(X_negative)
        # print("path lengths of positive samples:\n", pathlengths_positive, len(pathlengths_positive))
        # print("path lengths of negative samples:\n", pathlengths_negative, len(pathlengths_negative))
        pathlength_stop = time.time()
        pathlength_time = pathlength_stop - pathlength_start
        # print(f"path length time {pathlength_time:3.2f}s")

        score_start = time.time()
        scores_positive = wpit.anomaly_score(X_positive)
        scores_negative = wnit.anomaly_score(X_negative)
        # print("anomaly scores of positive samples:\n", scores_positive, len(scores_positive))
        # print("anomaly scores of negative samples:\n", scores_negative, len(scores_negative))
        score_stop = time.time()
        score_time = score_stop - score_start
        # print(f"score time {score_time:3.2f}s")

        y_predict1_positive = wpit.predict_threshold(X_positive, threshold)
        y_predict2_positive = wpit.predict_percentile(X_positive, percentile)

        y_predict1_negative = wnit.predict_threshold(X_negative, threshold)
        y_predict2_negative = wnit.predict_percentile(X_negative, percentile)
        # print(y_predict1, '\n', y_predict2)

        # print(X_positive_idx_list, X_negative_idx_list)
        left_X_positive, left_y = [], []
        left_idx = []
        for i in range(len(y_predict2_positive)):
            if y_predict2_positive[i] == 0:
                left_X_positive.append(X_positive[i])
                left_y.append(1)
                left_idx.append(X_positive_idx_list[i])   # note: the original index in X_Smote

        left_X_negative = []
        for i in range(len(y_predict2_negative)):
            if y_predict2_negative[i] == 0:
                left_X_negative.append(X_negative[i])
                left_y.extend([0])                         # add an element in a list instead of adding a list in a list
                left_idx.extend([X_negative_idx_list[i]])    # note: the original index in X_Smote

        left_X_positive = np.array(left_X_positive)
        left_y = np.array(left_y)
        left_X_negative = np.array(left_X_negative)

        left_X = np.vstack((left_X_positive, left_X_negative))
        # print(left_idx, '\n', left_X, type(left_X), left_X.shape, '\n', left_y, type(left_y), left_y.shape)

        # left_Data = np.c_[left_X, left_y]
        # print("left Data: \n", left_Data, left_Data.shape, type(left_Data))

        return left_idx, left_X, left_y


if __name__ == '__main__':
    np.random.seed(21)
    random.seed(21)
    r = 5
    # print(c(0), c(1), c(2), c(3), '\n',  c(4), c(5), c(9))
    datafile = 'cancer.csv'
    df = pd.read_csv(datafile)
    targetcol = 'diagnosis'
    sample_size = int(8)
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
    left_idx,  left_X, left_y = detect_anomalies_improvedplusLabel(DATA, random_state=r + 1, sample_size=sample_size, n_trees=n_trees, threshold=0.5, percentile=0.2)
    print("wif+l", left_idx,  left_y, len(left_y)) # left_X,
