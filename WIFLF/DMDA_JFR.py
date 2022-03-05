"""
Joint Feature Representation with Double Marginalized Denoising Autoencoders (DMDA_JFR):
    REFERENCE:
    Zou, Quanyi, Lu Lu, Zhanyu Yang, Xiaowei Gu, and Shaojian Qiu. "Joint feature representation learning and progressive
    distribution matching for cross-project defect prediction." Information and Software Technology 137 (2021): 106588.

"""
import numpy as np
import os
import pandas as pd
import time
import copy
from Performance import performance
from scipy import optimize, sparse
from sklearn import preprocessing
from scipy.linalg import norm, block_diag
from sklearn import linear_model, neighbors, svm, naive_bayes, tree, ensemble, neural_network
from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Standard_Features, Log_transformation  # Process_Data,
from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model


def DMDA_JFR(X_src,Y_src, X_tar, parameter, r, classifier):
    """
    FUNCTION DMDA_JFR() SUMMARY GOES HERE:
    # Step1: joint feature representation learning
      We utilize two novel auto-encoders to jointly learn the global and local feature representations simultaneously.
    # Step2: progressive distribution  matching
      To achieve progressive distribution matching, we introduce a repetitious pseudo-labels strategy, which makes it
    possible that distributions are matched after each stack layer learning rather than in one stroke.
    :param X_src: source feature matrix, n_src * m
    :param Y_src: source label vector, n_src * 1
    :param X_tar: target feature matrix, n_tar * m
    :param parameter: type: dict, {"noises": p, "layers": L,  "lambda": lmd, "beta":beta},  #
                      Number of the stacked layers L, Noise probability (corruption level) p, Regularization parameter
                      lamda, trade-off parameter beta.
    :param r: random state, int
    :param classifier: name of classifier, string, "LR"(Default), "NB", "RF", "SVM", “KNN”
    :return: Y_tar_pseudo: the predict labels of the target project,  n_tar * 1.
    """

    Y_tar_pseudo = Pseudolabel(X_src, X_tar, Y_src, r, classifier)

    for l in range(parameter["layers"]):
        if len(np.unique(Y_tar_pseudo)) == 2:
            num_sub_src, sub_src = subData(X_src, Y_src)
            num_sub_tar, sub_tar = subData(X_tar, Y_tar_pseudo)
            label = np.unique(Y_src)
            Num_Class = len(label)
            # // DA-GMDA stage
            # Construct matrix 𝜦 with Eq. (11);
            # Obtain the mapping matrix 𝐖 with Eq. (10);
            # Obtain the source global feature presentations 𝐗𝑙𝑠 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠 );
            # Obtain the target global feature presentations 𝐗𝑙𝑡= 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡 );
            X_src_globalx, X_tar_globalx, W = DA_GMDA(X_src, Y_src, X_tar, Y_tar_pseudo, parameter)
            # Y_tar_pseudo = Pseudolabel(X_src_globalx, X_tar_globalx, Y_src)  # Update pseudo label

            # // DA-LMDA stage
            X_src_localx = []
            X_tar_localx = []
            for c in range(Num_Class):
                # Construct matrix 𝜦𝑐 with Eq. (14);
                # Obtain the local subset mapping matrix 𝐖𝑐 with Eq. (13);
                # Obtain the source local feature presentations 𝐗𝑙𝑠,𝑐 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠,𝑐 );
                # Obtain the target local feature presentations 𝐗𝑙𝑡,𝑐 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡,𝑐 );

                X_src_localxc, X_tar_localxc, Wc = DA_LMDA(Xc_src=sub_src[c], Yc_src=c * np.ones((num_sub_src[c], 1)),
                                                           Xc_tar=sub_tar[c], Yc_tar_pseudo=Y_tar_pseudo,
                                                           parameter=parameter)
                X_src_localx.append(X_src_localxc)
                X_tar_localx.append(X_tar_localxc)

                # Y_tar_pseudo = Pseudolabel(X_src_localx, X_tar_localx, Y_src)  # Update pseudo label

            X_src_localx = np.concatenate((X_src_localx[0], X_src_localx[1]), axis=0)
            X_tar_localx = np.concatenate((X_tar_localx[0], X_tar_localx[1]), axis=0)

            # The source features presentations [𝐗𝑙𝑠,𝐗𝑙𝑠,𝑐1 ,𝐗𝑙𝑠,𝑐2 ];
            X_src_new = np.concatenate((X_src_globalx, X_src_localx),
                                       axis=1)  # X_src_new: source joint feature representations
            # The target features presentations [𝐗𝑙𝑡,𝐗𝑙𝑡,𝑐1 ,𝐗𝑙𝑡,𝑐2 ];
            X_tar_new = np.concatenate((X_tar_globalx, X_tar_localx),
                                       axis=1)  # X_tar_new: target joint feature representations

            # Train a classifier 𝑓(⋅) by using the source features presentations and labels of source project;
            Y_tar_pseudo_New = Pseudolabel(X_src_new, X_tar_new, Y_src, r, classifier)

            if len(np.unique(Y_tar_pseudo)) == 2:
                Y_tar_pseudo = Y_tar_pseudo_New
            else:
                pass
        else:
            Y_tar_pseudo = Pseudolabel(X_src, X_tar, Y_src, r+100, classifier)


    return Y_tar_pseudo

def DA_GMDA(X_src,Y_src,X_tar,Y_tar_pseudo, parameter):
    """
    FUNC DA_GMDA() SUMMARY GOES HERE:
    Generate transferable global.
    DA-GMDA leverages both 𝐷𝑠 and 𝐷𝑡 to extract the global features with more transferable capacity. This work assumes
    that 𝑃(𝐱𝑠) ≠ 𝑃(𝐱𝑡) and 𝑃(𝑦𝑠∣𝐱𝑠) ≠ 𝑃(𝑦𝑡∣𝐱𝑡) so that marginal and conditional probability distributions are matched
    simultaneously. DA-GMDA aims to learn the mapping matrix 𝐖 to reconstruct the transferable global feature space.

    :param X_src: source feature matrix, n_src * m
    :param Y_src: source label vector, n_src * 1
    :param X_tar: target feature matrix, n_tar * m
    :param Y_tar_pseudo: target pseudo-label vector, n_tar * 1
    :param parameter: option dict, parameter= {"noises": p,"layers": L,"lambda": lmd, "beta":beta}
    :return: X_src_globalx: source global feature representations;
             X_tar_globalx: target global feature representations
             W: global feature mapping matrix.
    """

    # Construct matrix 𝜦 with Eq. (11);
    n_src = X_src.shape[0]  # instance number
    n_tar = X_tar.shape[0]  # instance number
    X = np.concatenate((X_src.T, X_tar.T), axis=1)  # concat in columns
    m = X.shape[0]  # row number: feature number
    n = X.shape[1]  # column number: instance number
    if n != (n_src + n_tar):
        print("Concact matrix is wrong, please check!")
    label = np.unique(Y_src)
    Num_Class = len(label)
    # Num_Class = len(np.unique(Y_src))

    # X = X * np.diag(sparse(1. / np.sqrt(np.sum(np.square(X), 1))))
    X_normalize = preprocessing.normalize(X, axis=0, norm='l2')
    # X = np.c_(X_normalize,  np.ones((1, n)))
    # X = np.insert(X_normalize, -1, values=np.ones((1, n)), axis=0)
    X = np.row_stack((X_normalize, np.ones((1, n))))  # adding bias
    Hsm = np.multiply(1 / np.square(n_src), (np.ones((n_src, 1)) * np.ones((n_src, 1)).T))
    Htm = np.multiply(1 / np.square(n_tar), (np.ones((n_tar, 1)) * np.ones((n_tar, 1)).T))
    Hstm = np.multiply(1 / (n_tar * n_src), (np.ones((n_src, 1)) * np.ones((n_tar, 1)).T))  # ? -

    num_sub_src, sub_src = subData(X_src, Y_src)
    num_sub_tar, sub_tar = subData(X_tar, Y_tar_pseudo)

    tempHsc = []
    tempHtc = []
    tempHstc = []
    for i in range(Num_Class):
        n_src_c = num_sub_src[i]  # ns,c
        Hsc1 = np.multiply(1 / np.square(n_src_c), (np.ones((n_src_c, 1)) * np.ones((n_src_c, 1)).T))
        n_tar_c = num_sub_tar[i]  # nt,c
        Htc1 = np.multiply(1 / np.square(n_tar_c), (np.ones((n_tar_c, 1)) * np.ones((n_tar_c, 1)).T))
        Hstc1 = np.multiply(1 / (n_tar_c * n_src_c), (np.ones((n_src_c, 1)) * np.ones((n_tar_c, 1)).T))  # ?-

        tempHsc.append(Hsc1)
        tempHtc.append(Htc1)
        tempHstc.append(Hstc1)
    Hsc = block_diag(tempHsc[0], tempHsc[1])
    Htc = block_diag(tempHtc[0], tempHtc[1])
    Hstc = block_diag(tempHstc[0], tempHstc[1])


    # Mss = np.dot(np.dot(X_src.T, (Hsm + Hsc)), X_src)  #
    # Mst = np.dot(np.dot(X_src.T, (Hstc + Hstm)), X_tar)  #
    # Mtt = np.dot(np.dot(X_tar.T, (Htm + Htc)), X_tar)  #X_tar * (Htm + Htc) * X_tar.T  # ? in the original paper is wrong: X_tar.T * (Hsm + Hsc) * X_tar
    #
    # A = np.concatenate((np.concatenate((Mss, -Mst), axis=1), np.concatenate((-Mst, Mtt), axis=1)), axis=0)
    # A = A / np.sqrt(np.sum(np.diag(A.T * A)))  # normalize A, the Frobenius norm, sqrt(sum(diag(X'*X))).
    # S2 = np.dot(A, A.T)  # A.T * A   # MMD

    M = np.concatenate((np.concatenate(((Hsm + Hsc), -(Hstc + Hstm)), axis=1),
                        np.concatenate((-(Hstc.T + Hstm.T), (Htm + Htc)), axis=1)), axis=0)
    M = M / np.sqrt(np.sum(np.diag(M.T * M)))
    S2 = np.dot(np.dot(X, M), X.T)  # MMD
    S1 = np.dot(X, X.T)  # or X * X.T  # scatter matrix

    # Obtain the mapping matrix 𝐖 with Eq. (10);
    q = np.ones((m+1, 1)) * (1-parameter["noises"])
    q[-1] = 1
    EQ1 = np.multiply(S1, (q * q.T))
    np.fill_diagonal(EQ1, np.multiply(q, np.diag(S1)))
    # EQ1[1:m + 2:end] = np.multiply(q, np.diag(S1))  # diag 对角线
    EQ2 = np.multiply(S2, (q * q.T))
    np.fill_diagonal(EQ2, np.multiply(q, np.diag(S2)))
    # EQ2[1:m + 2:end] = np.multiply(q, np.diag(S2))
    EP = np.multiply(q, S1)  #  EP = np.multiply(Sc[:-1, :], repmat(q.T, m, 1))  # m*(m+1)
    reg = parameter["lambda"] * np.eye((m + 1))
    reg[-1, -1] = 0

    W = EP / (EQ1 + reg + parameter["beta"] * EQ2)  # global feature  matrix mapping??"beta"

    # Obtain the source global feature presentations 𝐗𝑙𝑠 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠);
    global_all = np.dot(W, X) #?? W*X.T
    global_all = np.tanh(global_all)
    global_all = preprocessing.normalize(global_all, axis=0, norm='l2')
    # global_all = global_all * np.diag(sparse(1. / sqrt(np.sum(np.square(global_all)))))  # ?? WHY?
    X_src_globalx = global_all[:, :n_src].T  # source global feature representations


    # Obtain the target global feature presentations 𝐗𝑙𝑡= 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡);
    X_tar_globalx = global_all[:, n_src:].T  # target global feature representations

    # Y_tar_pseudo = Pseudolabel(X_src_globalx, X_tar_globalx, Y_src)  # Update pseudo label
    # X = global_all

    return X_src_globalx, X_tar_globalx, W

def DA_LMDA(Xc_src, Yc_src, Xc_tar, Yc_tar_pseudo, parameter):
    """
    FUNC DA_LMDA() SUMMARY GOES HERE:
    Generate transferable local feature.
    :param Xc_src: source feature matrix, n_src * m, 2D array;
    :param Yc_src: source label vector, n_src * 1, 1D array;
    :param Xc_tar: target feature matrix, n_tar * m, 2D array;
    :param Y_tar_pseudo: target pseudo-label vector, n_tar * 1, 1D array;
    :param parameter: option dict, parameter= {"layers": , "noises": , "lambda": , "beta":}
    :return: X_src_localx: source local feature representations;
             X_tar_localx: target local feature representations;
             Wc: local feature mapping matrix.
    """

    # Construct matrix 𝜦𝑐 with Eq. (14);
    label = np.unique(Yc_src)
    # Num_Class = len(label)
    Xc = np.concatenate((Xc_src.T, Xc_tar.T), axis=1)  # concat in rows
    m = Xc.shape[0]  # row number: feature number
    nc = Xc.shape[1]  # column number: instance number
    n_src_c = len(Xc_src)
    n_tar_c = len(Xc_tar)
    if nc != (n_src_c + n_tar_c):
        print("Concact matrix is wrong, please check!")

    # 向量化（右乘对角矩阵）,采用右乘一个对角矩阵的方式对矩阵进行缩放，常见的列向量单位化操作。2-范数：║x║2=（│x1│2+│x2│2+…+│xn│2）1/2
    # Xc = Xc * np.diag(sparse(1. / np.sqrt(np.sum(np.square(Xc), 1))))
    Xc_normalize = preprocessing.normalize(Xc, axis=0, norm='l2')
    # Xc = np.concatenate((Xc_normalize, np.ones(1, nc)), axis=1)
    Xc = np.row_stack((Xc_normalize, np.ones((1, nc))))

    Hsc_c = np.multiply((1 / np.square(n_src_c)), (np.ones((n_src_c, 1)) * np.ones((n_src_c, 1)).T))

    Htc_c = np.multiply((1 / np.square(n_tar_c)), (np.ones((n_tar_c, 1)) * np.ones((n_tar_c, 1)).T))
    Hstc_c = np.multiply((1 / (n_tar_c * n_src_c)), (np.ones((n_src_c, 1)) * np.ones((n_tar_c, 1)).T))

    # Mss_c = np.dot(np.dot(Xc_src.T, Hsc_c), Xc_src)
    # Mst_c = np.dot(np.dot(Xc_src.T, Hstc_c), Xc_tar)
    # Mtt_c = np.dot(np.dot(Xc_tar.T, Htc_c), Xc_tar)

    # Ac = np.concatenate((np.concatenate((Mss_c, -Mst_c), axis=1), np.concatenate((-Mst_c, Mtt_c), axis=1)), axis=0)
    # Ac = Ac / np.sqrt(np.sum(np.diag(Ac.T * Ac)))  # normalize Ac
    # S2c = Ac * Ac.T  # MMD

    Mc = np.concatenate((np.concatenate((Hsc_c, -Hstc_c), axis=1), np.concatenate((-Hstc_c.T, Htc_c), axis=1)), axis=0)
    Mc = Mc / np.sqrt(np.sum(np.diag(Mc.T * Mc)))  # MMD
    S2c = np.dot(np.dot(Xc, Mc), Xc.T)

    S1c = np.dot(Xc, Xc.T)

    # Obtain the local subset mapping matrix 𝐖𝑐 with Eq. (13);
    q = np.ones((m + 1, 1)) * (1 - parameter["noises"])
    EQ1c = np.multiply(S1c, (q * q.T))
    np.fill_diagonal(EQ1c, np.multiply(q, np.diag(S1c)))   # diag 对角线
    EQ2c = np.multiply(S2c, (q * q.T))
    np.fill_diagonal(EQ2c, np.multiply(q, np.diag(S2c)))   # diag 对角线
    EPc = np.multiply(q, S1c)  # EPc = np.multiply(Sc[:-1, :], repmat(q.T, m, 1))  # m*(m+1)
    reg = parameter["lambda"] * np.eye((m + 1))
    reg[-1, -1] = 0
    Wc = EPc / (EQ1c + reg + parameter["beta"] * EQ2c)

    # Obtain the source local feature presentations 𝐗𝑙𝑠,𝑐 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠,𝑐 );
    local_all = np.dot(Wc, Xc)  # local feature  matrix mapping
    local_all = np.tanh(local_all)
    local_all = preprocessing.normalize(local_all, axis=0, norm='l2')
    # local_all = local_all * diag(sparse(1. / np.sqrt(np.sum(np.square(local_all)))))

    X_src_localx = local_all[:, :n_src_c].T  # source local feature representations

    # Obtain the target local feature presentations 𝐗𝑙𝑡,𝑐 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡,𝑐 );
    X_tar_localx = local_all[:, n_src_c:].T  # target local feature representations

    return X_src_localx, X_tar_localx, Wc

def Pseudolabel(src_X, tar_X, src_labels, r, classifier="LR"):
    """
    FUNCTION PSEUDOLABEL() SUMMARY GOES HERE:
    Obtain pseudo labels in target project which is trained by the source project with a classifier. In this paper,
    CPDP is regarded as binary so that the dependent variable is a binary variable 𝑐 ∈ {𝑐1, 𝑐2}.
    :param src_X: source feature matrix, n_src * m, 2D array;
    :param tar_X: target feature matrix, n_tar * m, 2D array;
    :param src_labels: source label vector, n_src * 1, 1D array;
    :return: Y_tar_pseudo: target pseudo-label vector, n_tar * 1, 1D array;
    """

    if classifier == "LR":
        # Logistic Regression classifier
        Y = src_labels
        # Y = (src_labels + 1) * 0.5 # [-1,1] vs[0,1]
        lr = linear_model.LogisticRegression(random_state=r + 1, solver='liblinear',
                                             max_iter=100, multi_class='ovr')
        lr.fit(src_X, Y)
        predlabel = lr.predict_proba(tar_X)  # predict probability

        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        for j in range(predlabel.shape[0]):  # 每个类的概率, array类型
            if len(lr.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if lr.classes_[1] == 1:
                    prob_pos.append(predlabel[j][1])
                    prob_neg.append(predlabel[j][0])
                else:
                    prob_pos.append(predlabel[j][0])
                    prob_neg.append(predlabel[j][1])
        prob_pos = np.array(prob_pos)
        Y_pseudo = (prob_pos >= 0.5).astype(np.int32)
        Y_tar_pseudo = Y_pseudo
        # Y_tar_pseudo = Y_pseudo * 2 - 1

    elif classifier == "KNN":
        # %% K-Nearest Neighbor (KNN) classifier
        knn = neighbors.KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                             metric_params=None, n_jobs=1, n_neighbors=1, p=2,
                                             weights='uniform')  # IB1 = knn IB1（KNN,K=1）
        knn.fit(src_X, src_labels)
        Y_tar_pseudo = knn.predict(tar_X)

        # knn_model=fitcknn(src_X,src_labels,'NumNeighbors',1)
        # Y_tar_pseudo=knn_model.predict(tar_X)

    elif classifier == "SVM":
        # SVM classifier
        xr = src_X.T  # xr instances * feature
        bestC = 1. / np.mean(np.sum(np.multiply(xr, xr), 2))
        svml = svm.SVC(C=1.0, kernel='linear', probability=True, random_state=r + 1,
                       gamma='auto', cache_size=3000)  # SVM + Linear  Kernel (SVML)
        svml.fit(xr, src_labels)
        xe = tar_X.T
        predlabel = svml.predict_proba(xe)  # predict probability
        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        for j in range(predlabel.shape[0]):  # 每个类的概率, array类型
            if len(svml.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if svml.classes_[1] == 1:
                    prob_pos.append(predlabel[j][1])
                    prob_neg.append(predlabel[j][0])
                else:
                    prob_pos.append(predlabel[j][0])
                    prob_neg.append(predlabel[j][1])
        prob_pos = np.array(prob_pos)
        Y_tar_pseudo = (prob_pos >= 0.5).astype(np.int32)

        #-q : quiet mode (no outputs); -t kernel_type : set type of kernel function (default 2),  0 – linear: u’v
        #-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1);
        #-m cachesize : set cache memory size in MB (default 100)
        # model = svmtrain(src_labels,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000'])
        # Y_tar_pseudo,accuracy = svmpredict(tar_labels,xe,model)

    elif classifier == "RF":
        # Random Forest (RF)
        rf = ensemble.RandomForestClassifier(random_state=r + 1, n_estimators=100)
        rf.fit(src_X, src_labels)
        predlabel = rf.predict_proba(tar_X)  # predict probability
        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        for j in range(predlabel.shape[0]):  # 每个类的概率, array类型
            if len(rf.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if rf.classes_[1] == 1:
                    prob_pos.append(predlabel[j][1])
                    prob_neg.append(predlabel[j][0])
                else:
                    prob_pos.append(predlabel[j][0])
                    prob_neg.append(predlabel[j][1])
        prob_pos = np.array(prob_pos)
        Y_tar_pseudo = (prob_pos >= 0.5).astype(np.int32)

    elif classifier == "NB":
        # Naive Bayes(NB)
        gnb = naive_bayes.GaussianNB(priors=None)
        gnb.fit(src_X, src_labels)
        predlabel = gnb.predict_proba(tar_X)  # predict probability
        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        for j in range(predlabel.shape[0]):  # 每个类的概率, array类型
            if len(gnb.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if gnb.classes_[1] == 1:
                    prob_pos.append(predlabel[j][1])
                    prob_neg.append(predlabel[j][0])
                else:
                    prob_pos.append(predlabel[j][0])
                    prob_neg.append(predlabel[j][1])
        prob_pos = np.array(prob_pos)
        Y_tar_pseudo = (prob_pos >= 0.5).astype(np.int32)

    else:
        # Logistic Regression classifier
        # Y = (src_labels + 1) * 0.5  # [-1,1] vs[0,1]
        Y = src_labels
        lr = linear_model.LogisticRegression(random_state=r + 1, solver='liblinear',
                                             max_iter=100, multi_class='ovr')
        lr.fit(src_X, Y)
        predlabel = lr.predict_proba(tar_X)  # predict probability
        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        for j in range(predlabel.shape[0]):  # 每个类的概率, array类型
            if len(lr.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if lr.classes_[1] == 1:
                    prob_pos.append(predlabel[j][1])
                    prob_neg.append(predlabel[j][0])
                else:
                    prob_pos.append(predlabel[j][0])
                    prob_neg.append(predlabel[j][1])
        prob_pos = np.array(prob_pos)
        Y_pseudo = (prob_pos >= 0.5).astype(np.int32)

        # model = glmfit(src_X,Y,'binomial', 'link', 'logit')
        # predlabel = glmval(model,tar_X, 'logit')
        # Y_pseudo=double(predlabel>=0.5)
        Y_tar_pseudo = Y_pseudo
        # Y_tar_pseudo = Y_pseudo * 2 - 1

    return Y_tar_pseudo

def subData(Xc, Y):
    """
    FUNCTION SUBDATA() SUMMARY GOES HERE:
    Divide the dataset into different local subsets (2 subsets: non-buggy instances and buggy instances) according to
    labels.
    :param Xc: 2D array, instances
    :param Y: 1D array, labels
    :return: num_sub: list,len=2, number of subsets, num_sub[0] is number of non-defective instances, while  num_sub[1]
                      is number of defective instances.
             sub_X: list, len=2, sub_X[0] (sub_X[1]) is 2D array and contains the non-defective (defective) instances.
    """

    num_sub = []  # [number of non-defective instances, number of defective instances]
    sub_X = []
    label = np.unique(Y)
    for i in range(len(label)):
        index = [j for j, x in enumerate(Y) if x == label[i]]
        num_sub.append(len(index))
        sub_X.append(Xc[index, :])

    return num_sub, sub_X

def DMDA_JFR_1(X_src, Y_src, X_tar, parameter, r, classifier):
    """
    FUNC DMDA_JFR_1() SUMMARY GOES HERE:
     The method only retains the joint feature representation and eliminates the repetitious pseudo-label strategy (all
     pseudo-labels remain unchanged in the whole learning process).
    :param X_src: source feature matrix, n_src * m
    :param Y_src: source label vector, n_src * 1
    :param X_tar: target feature matrix, n_tar * m
    :param parameter: type: dict, {"noises": p, "layers": L,  "lambda": lmd, "beta":beta},  #
                      Number of the stacked layers L, Noise probability (corruption level) p, Regularization parameter
                      lamda, trade-off parameter beta.
    :param r: random state, int
    :param classifier: name of classifier, string, "LR"(Default), "NB", "RF", "SVM", “KNN”
    :return: Y_tar_pseudo: the predict labels of the target project,  n_tar * 1.
    """

    Y_tar_pseudo = Pseudolabel(X_src, X_tar, Y_src, r, classifier)
    if len(np.unique(Y_tar_pseudo)) == 2:
        pass
    else:
        Y_tar_pseudo = Pseudolabel(X_src, X_tar, Y_src, r + 100, classifier)

    for l in range(parameter["layers"]):
        num_sub_src, sub_src = subData(X_src, Y_src)
        num_sub_tar, sub_tar = subData(X_tar, Y_tar_pseudo)
        label = np.unique(Y_src)
        Num_Class = len(label)
        # // DA-GMDA stage
        # Construct matrix 𝜦 with Eq. (11);
        # Obtain the mapping matrix 𝐖 with Eq. (10);
        # Obtain the source global feature presentations 𝐗𝑙𝑠 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠 );
        # Obtain the target global feature presentations 𝐗𝑙𝑡= 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡 );
        X_src_globalx, X_tar_globalx, W = DA_GMDA(X_src, Y_src, X_tar, Y_tar_pseudo, parameter)
        # Y_tar_pseudo = Pseudolabel(X_src_globalx, X_tar_globalx, Y_src)  # Update pseudo label

        # // DA-LMDA stage
        X_src_localx = []
        X_tar_localx = []
        for c in range(Num_Class):
            # Construct matrix 𝜦𝑐 with Eq. (14);
            # Obtain the local subset mapping matrix 𝐖𝑐 with Eq. (13);
            # Obtain the source local feature presentations 𝐗𝑙𝑠,𝑐 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠,𝑐 );
            # Obtain the target local feature presentations 𝐗𝑙𝑡,𝑐 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡,𝑐 );

            X_src_localxc, X_tar_localxc, Wc = DA_LMDA(Xc_src=sub_src[c], Yc_src=c * np.ones((num_sub_src[c], 1)),
                                                       Xc_tar=sub_tar[c], Yc_tar_pseudo=Y_tar_pseudo,
                                                       parameter=parameter)
            X_src_localx.append(X_src_localxc)
            X_tar_localx.append(X_tar_localxc)

            # Y_tar_pseudo = Pseudolabel(X_src_localx, X_tar_localx, Y_src)  # Update pseudo label

        X_src_localx = np.concatenate((X_src_localx[0], X_src_localx[1]), axis=0)
        X_tar_localx = np.concatenate((X_tar_localx[0], X_tar_localx[1]), axis=0)

        # The source features presentations [𝐗𝑙𝑠,𝐗𝑙𝑠,𝑐1 ,𝐗𝑙𝑠,𝑐2 ];
        X_src_new = np.concatenate((X_src_globalx, X_src_localx),
                                   axis=1)  # X_src_new: source joint feature representations
        # The target features presentations [𝐗𝑙𝑡,𝐗𝑙𝑡,𝑐1 ,𝐗𝑙𝑡,𝑐2 ];
        X_tar_new = np.concatenate((X_tar_globalx, X_tar_localx),
                                   axis=1)  # X_tar_new: target joint feature representations

    X_src = X_src_new
    X_tar = X_tar_new
    Y_tar_pseudo = Pseudolabel(X_src, X_tar, Y_src, r + 100, classifier)

    return Y_tar_pseudo

def DMDA_JFR_2(X_src, Y_src, X_tar, parameter, r, classifier):
    """
    FUNC DMDA_JFR_2() SUMMARY GOES HERE:
    The method has only the repetitious pseudo-labels strategy without the joint feature representation.
    :param X_src: source feature matrix, n_src * m
    :param Y_src: source label vector, n_src * 1
    :param X_tar: target feature matrix, n_tar * m
    :param parameter: type: dict, {"noises": p, "layers": L,  "lambda": lmd, "beta":beta},  #
                      Number of the stacked layers L, Noise probability (corruption level) p, Regularization parameter
                      lamda, trade-off parameter beta.
    :param r: random state, int
    :param classifier: name of classifier, string, "LR"(Default), "NB", "RF", "SVM", “KNN”
    :return: Y_tar_pseudo: the predict labels of the target project,  n_tar * 1.
    """
    Y_tar_pseudo = Pseudolabel(X_src, X_tar, Y_src, r, classifier)

    for l in range(parameter["layers"]):
        if len(np.unique(Y_tar_pseudo)) == 2:
            pass
        else:
            Y_tar_pseudo = Pseudolabel(X_src, X_tar, Y_src, r + 100, classifier)

        # // DA-GMDA stage
        # Construct matrix 𝜦 with Eq. (11);
        # Obtain the mapping matrix 𝐖 with Eq. (10);
        # Obtain the source global feature presentations 𝐗𝑙𝑠 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠 );
        # Obtain the target global feature presentations 𝐗𝑙𝑡= 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡 );
        X_src_globalx, X_tar_globalx, W = DA_GMDA(X_src, Y_src, X_tar, Y_tar_pseudo, parameter)

        Y_tar_pseudo = Pseudolabel(X_src_globalx, X_tar_globalx, Y_src, r + 100, classifier)  # Update pseudo label


    return Y_tar_pseudo


def DMDA_JFR_main(mode, clf_index, runtimes):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    datasets = [
        ['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv',
         'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv',
         'xalan-2.4.csv', 'xerces-init.csv'],
        ['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv', 'lucene-2.0.csv',
         'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv',
         'xalan-2.6.csv', 'xerces-1.3.csv'],
        ['EQ.csv', 'JDT.csv', 'LC.csv', 'ML.csv', 'PDE.csv'],
        ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    parameters = [[{"layers": 4, "noises": 0.5, "lambda": 0.01, "beta": 0.1},  # Ant-1.3 [4, 0.5, 10−2]
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # arc-1.
                   {"layers": 4, "noises": 0.7, "lambda": 0.1, "beta": 0.1},  # Camel-1.0 [4, 0.7, 10−1]
                   {"layers": 5, "noises": 0.6, "lambda": 0.1, "beta": 0.1},  # Ivy-1.4 [5, 0.3, 10 ]
                   {"layers": 8, "noises": 0.6, "lambda": 0.1, "beta": 0.1},  # Jedit-3.2 [8, 0.6, 10−1 ]
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # log4j-1.0
                   {"layers": 8, "noises": 0.4, "lambda": 1000, "beta": 0.1},  # Lucene-2.0 [8, 0.4, 103 ]
                   {"layers": 5, "noises": 0.4, "lambda": 0.01, "beta": 0.1},  # Pio-2.0 [5, 0.4, 10−2 ]
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # redaktor-1
                   {"layers": 5, "noises": 0.9, "lambda": 0.01, "beta": 0.1},  # Synapse-1.0 [5, 0.9, 10−2]
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # tomcat-6.0.
                   {"layers": 8, "noises": 0.8, "lambda": 10, "beta": 0.1},  # Velocity-1.6 [ 8, 0.8, 10]
                   {"layers": 5, "noises": 0.7, "lambda": 0.1, "beta": 0.1},  # Xalan-2.4 [5, 0.7, 10−1]
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1}],  # xerces-init
                  [{"layers": 8, "noises": 0.3, "lambda": 0.001, "beta": 0.1},  # Ant-1.7 [8, 0.3, 10−3]
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # arc-1.
                   {"layers": 6, "noises": 0.6, "lambda": 1, "beta": 0.1},  # Camel-1.6 [ 6, 0.6, 1]
                   {"layers": 6, "noises": 0.6, "lambda": 1, "beta": 0.1},  # Ivy-2.0 [6, 0.6, 1]
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # jedit-4.3
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # log4j-1.1
                   {"layers": 8, "noises": 0.4, "lambda": 1000, "beta": 0.1},  # Lucene-2.0 [8, 0.4, 103 ]
                   {"layers": 5, "noises": 0.4, "lambda": 0.01, "beta": 0.1},  # Pio-2.0 [5, 0.4, 10−2 ]
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # redaktor-1
                   {"layers": 8, "noises": 0.7, "lambda": 0.01, "beta": 0.1},  # Synapse-1.2 [8, 0.7, 10−2]
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # tomcat-6.0.
                   {"layers": 8, "noises": 0.8, "lambda": 10, "beta": 0.1},  # Velocity-1.6 [ 8, 0.8, 10]
                   {"layers": 5, "noises": 0.6, "lambda": 10, "beta": 0.1},  # Xalan-2.6 [5, 0.6, 10 ]
                   {"layers": 8, "noises": 0.2, "lambda": 100, "beta": 0.1}],  # Xerces-1.3 [8, 0.2, 102]
                  [{"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # EQ
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # JDT
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # LC.
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # ML
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1}],  # PDE
                  [{"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # Apache
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # Safe
                   {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1}  # Zxing
                   ]]
      # # example
   # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]
   # parameters = [[{"layers": 4, "noises": 0.5, "lambda": 0.01, "beta": 0.1},  # Ant-1.3 [4, 0.5, 10−2]
   #                {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # arc-1.
   #                {"layers": 4, "noises": 0.7, "lambda": 0.1, "beta": 0.1}],  # Camel-1.0 [4, 0.7, 10−1]
   #               [{"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # Apache
   #                {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},  # Safe
   #                {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1}  # Zxing
   #                ]]

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
        m = -1
        for file_te in datasets[i]:
            m += 1
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
                            print('train_file:', file_tr)
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

                    # /* step2: Zscore/
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore', X_test)  # data transformation

                    # /*step3: DMDA_JFR*/
                    modelname, classifier = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname)
                    # y_predict = DMDA_JFR(X_train, y_train, X_test, parameter, r, classifier=modelname)
                    y_predict = DMDA_JFR(X_train_zscore, y_train, X_test_zscore, parameters[i][m], r, classifier=modelname)
                    measures_dmdajfr = performance(y_test, y_predict)

                    end_time = time.time()
                    run_time = end_time - start_time
                    measures_dmdajfr.update(
                        {'train_len_before': len(X_train), 'train_len_after': len(X_train), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_dmdajfr, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
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


if __name__ == '__main__':

    mode = ['null', 'M2O_CPDP', 'eg_DMDA_JFR']
    for i in range(0, 3, 1):
        DMDA_JFR_main(mode, clf_index=i, runtimes=3)





