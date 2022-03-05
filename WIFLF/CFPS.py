
"""
REFERENCE:
    Sun, Z., Li, J., Sun, H., He, L., 2021. CFPS: Collaborative filtering based source projects selection for cross-
    project defect prediction. APPLIED SOFT COMPUTING 99. doi:{10.1016/j.asoc.2020.106940}.

Q1: why simScores always the same for the same target?
(A1: From cc‘s code, we use z-score for NP and HPs, so the mean is close to 0, std is 1, so the NP-HPi is always the same.
So, we change the order of CFPS and Z-score.)
Q2: which measureString should be selected? 'AUC' OR 'F1'?
Q3: Which learner should be selected?  # RF, Naive Bayes (NB),Logistic Regression (LR), Decision Tree (C4.5), Sequential Minimal Optimization (SMO).
(Q2+Q3: select "F1+RF": In the original paper, RF and SMO seem to be improved most significantly.)
Q4: Should set K as which number? K=3,5,7, 10.
(For PROMISE, SET K=10, Because K=10, the CFPS got the best performance.
But for AEEEM only 5 datasets set K=3; For Relink only 3 datasets set K =1)
###Q5: In the original paper, the authors only pay attention to the recommend performance instead of using these
recommended source projects to train an effective prediction model.###
"""
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist

from sklearn import naive_bayes, ensemble, svm
import pandas as pd
import time
import os
from copy import deepcopy
from sklearn import metrics

from Performance import performance
from collections import Counter

# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler

from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Standard_Features, Log_transformation  # Process_Data,
from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model

def CFPS(HPs, NP, clf_index, r, measureString, K):
    """
    FUNCTION CFPS() SUMMARY GOES HERE:
    A Collaborative Filtering based source Projects Selection (CFPS) method: M2O-CPDP (K)
    :param HPs, type: list, each element in HPs is 2D array type. Historical Projects= {HP1, HP2, · · · , HPM}, all
                available historical projects with N attributes and 1 label.
    :param NP, type: 2D array. New Project, a new project with N attributes and 1 label.
    :param clf_index, type: int. the index of the learner.
    :param r, type: int. random state.
    :param: K, type: int, the number of selected applicable source projects.
    :return: ASPs, type: 2D array, applicable source projects for the new/target project.
    """
    # step1: Inter-project similarity mining
    SimiScores, simIndexArray = GetSimiScores(HPs, NP)

    # step2: Inter-Project Applicability Mining
    AppScores, appIndexArray = GetAppScores(HPs, clf_index, r, measureString)
    # step3: Collaborative Filtering Recommendation
    RecScores, recIndexArray = GetRecScores(SimiScores, AppScores)

    print(simIndexArray, appIndexArray, recIndexArray)
    selectedIdx = recIndexArray.tolist()
    ASPs = []  # Selected K applicable source projects for the new/target project
    for i in range(K):
        ASPs.append(HPs[selectedIdx[i]])
    return ASPs

def GetSimiScores(HPs, NP):
    """
    FUNCTION GETSIMISCORES() SUMMARY GOES HERE:
    Inter-project similarity mining: for a given new or target project, the similarity relationships between it and each
    of the historical projects are explored and we then record the corresponding similarity scores into the similarity
    repository.
    :param HPs: type: list, each element in HPs is 2D array type. Historical Projects= {HP1, HP2, · · · , HPM}, all
                available historical projects with N attributes and 1 label.
    :param NP: type: 2D array. New Project, a new project with N attributes and 1 label.
    :return: SimiScores: type: 1D array. Similarity Scores, a similarity score represents the similar degree between the
             new project and a historical project.
    """

    AVG_N_NP = NP.mean(axis=0)
    SD_N_NP = NP.std(axis=0)
    MetaFeatureVector_NP = np.concatenate((AVG_N_NP[:-1], SD_N_NP[:-1]))  # obtain the meta-feature vector for NP
    # print(MetaFeatureVector_NP)
    M = len(HPs)
    Euc = np.zeros((M,))
    SimiScores = np.zeros((M,))
    MetaFeatureVector_HPs = []
    for i in range(M):
        AVG_N_HPi = HPs[i].mean(axis=0)  # Calculate the average value of each attribute in HPi
        SD_N_HPi = HPs[i].std(axis=0)  # Calculate the average value of each attribute in HPi
        MetaFeatureVector_HPi = np.concatenate((AVG_N_HPi[:-1], SD_N_HPi[:-1]))  # Construct a meta-feature vector for HPi
        # print(MetaFeatureVector_HPi)
        MetaFeatureVector_HPs.append(MetaFeatureVector_HPi)
        # Calculate the Euclidean distance Euc[i] between MetaFeatureVector_HPi and MetaFeatureVector_NP
        Euc[i] = euclidean(MetaFeatureVector_NP, MetaFeatureVector_HPi)

        # X = np.vstack([MetaFeatureVector_HPi, MetaFeatureVector_NP])
        # # print(X[0], X[1])
        # Euc[i] = pdist(X)

        # subFeatureVector = np.subtract(MetaFeatureVector_HPi, MetaFeatureVector_NP)
        # subFeatureVector = MetaFeatureVector_HPi - MetaFeatureVector_NP
        # squSumFeatureVector = np.sum(np.square(subFeatureVector))
        # Euc[i] = np.sqrt(squSumFeatureVector)
        # print("%d,%f" % (i, Euc[i]))

        SimiScores[i] = 1/(1+Euc[i])  # Calculate the similarity score SimiScores[i]

    simIndexArray = np.argsort(-SimiScores)  # rank from max to min, return the index of SimiScores
    return SimiScores, simIndexArray


def GetAppScores(HPs, clf_index, r, measureString):
    """
    FUNCTION GETAPPSCORES() SUMMARY GOES HERE:
    Inter-project applicability mining: the applicabilities for each pair of historical projects are investigated
    empirically. Then an applicability repository is employed to preserve the corresponding applicability scores.
    :param HPs: Historical Projects= {HP1, HP2, · · · , HPM}, All available historical projects with N attributes.
    :param clf_index,
    :param r,
    :param measureString, type:string, "AUC" or "F1"
    :return: AppScores: Applicability Scores, a applicability score represents the applicability for each pair of
             historical projects and it is unsymmetrical.
    """
    AppScores = np.zeros((len(HPs), len(HPs)))
    for i in range(len(HPs)):  # |HPs| = M
        HPi = HPs[i]
        for j in range(len(HPs)):  #
            # TPs = deepcopy(HPs)
            # TPs.remove(TPs[i])  # TrainingProjects, |TPs| = M-1
            if j != i:
                HPj = HPs[j]
                # learner: such as Random Forest [54] and Naive Bayes [55].
                # in the original paper: Performance: F-Measure and AUC
                try:
                    measuresDict = learnerTrainTest(clf_index, r, HPj,
                                                    HPi)  # self-defined func, train HPj and predict HPi and record the performance.
                    AppScores[i][j] = measuresDict[measureString]
                except (IOError, ZeroDivisionError):
                    AppScores[i][j] = 0
                else:
                    measuresDict = learnerTrainTest(clf_index, r, HPj, HPi)
                    AppScores[i][j] = measuresDict[measureString]
                # finally:
                #     print("prediction performance is done!")

    appIndexArray = np.argsort(-AppScores)

    return AppScores, appIndexArray


def GetRecScores(SimiScores, AppScores):
    """
    FUNCITON GETRECSCORES() SUMMARY GOES HERE:
    Collaborative filtering recommendation: with the above obtained similarity and applicability repositoryThe user-
    based collaborative filtering recommendation algorithm[32] is employed to obtain recommended scores and recommend
    the appropriate source projects for the new project.
    REFERENCE: [32]"G. Adomavicius, A. Tuzhilin, Toward the next generation of recommender systems: A survey of the
               state-of-the-art and possible extensions, IEEE Trans. Knowl. Data Eng. 17 (6) (2005) 734–749."
    :param SimiScores: a 1*M array from Similarity Repository
    :param AppScores: a M*M array from Applicability Repository
    :return: RecScores: Recommended Scores;
    """
    M = len(SimiScores)
    RecScores = np.zeros((M,))
    for i in range(M):
        for j in range(M):
            RecScores[i] += SimiScores[j] * AppScores[j][i]  # Calculate the recommended score for the source project HPi

    recIndexArray = np.argsort(-RecScores)

    return RecScores, recIndexArray


def learnerSelection(clf_index, r):

    rf = ensemble.RandomForestClassifier(random_state=r + 1, n_estimators=100)  # Random Forest (RF)
    gnb = naive_bayes.GaussianNB(priors=None)  # Naive Bayes (NB)
    # Logistic Regression (LR)
    # Decision Tree (C4.5)
    # Sequential Minimal Optimization (SMO).
    clfs = [{'NB': gnb}, {'RF': rf}]
    classifier = clfs[clf_index]
    modelname = str(list(classifier.keys())[0])  # name of the learner
    clf = list(classifier.values())[0]  # model of the learner

    return modelname, clf

def learnerTrainTest(clf_index, r, trainData, testData):

    X_train = trainData[:, :-1]
    y_train = trainData[:, -1]
    X_test = testData[:, :-1]
    y_test = testData[:, -1]
    classifierName, classifier = learnerSelection(clf_index, r)
    # print(classifierName)
    classifier.fit(X_train, y_train)
    # y_te_prepredict = classifier.predict(X_test)  # predict labels
    y_te_predict_proba = classifier.predict_proba(X_test)  # predict probability
    prob_pos = []  # the probability of predicting as positive
    prob_neg = list()  # the probability of predicting as negative
    # print(X_train, X_train.shape, y_train, y_train.shape,
    #       X_test, X_test.shape, y_test, y_test.shape, classifier.classes_)
    for j in range(y_te_predict_proba.shape[0]):  # 每个类的概率, array类型
        if len(classifier.classes_) <= 1:
            print('第%d个样本预测出错' % j)
        else:
            if classifier.classes_[1] == 1:
                prob_pos.append(y_te_predict_proba[j][1])
                prob_neg.append(y_te_predict_proba[j][0])
            else:
                prob_pos.append(y_te_predict_proba[j][0])
                prob_neg.append(y_te_predict_proba[j][1])
    prob_pos = np.array(prob_pos)
    prob_neg = np.array(prob_neg)
    measures = predicitonPerformance(y_test, prob_pos)

    return measures

def predicitonPerformance(y_true, prob_pos):
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # sklearn.metrics.confusion_matrix
    :param y_true:  A 1-D array that denotes test instances' actual labels, {0,1} where 1 is defective, 0 non-defective.
    :param probPos: A 1-D array, which denotes the probability of predicted being defective for each instance.
    :return: A dict including multiple elements, such as F1-Measure, AUC.
    """
    assert (np.unique(y_true) > 1, 'Please ensure that ''y_true'' includes two or more different labels.')  #
    assert (len(y_true) == len(prob_pos), 'Two input parameters must have the same size.')
    y_pred = (prob_pos >= 0.5).astype(np.int32)
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()

    try:
        F1 = metrics.f1_score(y_true, y_pred)
        AUC = metrics.roc_auc_score(y_true, prob_pos)
    except (IOError, ZeroDivisionError):
        F1 = 0
        AUC = 0
    else:
        F1 = metrics.f1_score(y_true, y_pred)
        AUC = metrics.roc_auc_score(y_true, prob_pos)

    measures = {'AUC': AUC, 'F1': F1}

    return measures


def recommendationPerformance(appList, recList, K, N, M):
    """
    three information retrieval evaluation metrics: F-Measure@N, MAP and MRR[60,61].
    REFERENCES:
    [60] R. Baeza-Yates, B.d.A.N. Ribeiro, et al., Modern Information Retrieval, ACM Press, New York, 2011.
    [61] T. Qin, T.-Y. Liu, J. Xu, H. Li, LETOR: A benchmark collection for research on learning to rank for information
        retrieval, Inf. Retr. 13 (4) (2010) 346–374.
    :param y_true:  A 1-D array that denotes test instances' actual labels, {0,1} where 1 is defective, 0 non-defective.
    :param probPos: A 1-D array, which denotes the probability of predicted being defective for each instance.
    :param K: the number of true applicable source projects, set K as the value 3, 5, 7, 10.
    :param N: the number of recommended source projects by the CPFS method, N is set the same value as K.
    :param M: the number of new projects to be recommended.
    :return:
    """

    # assert (np.unique(y_true) > 1, 'Please ensure that ''y_true'' includes two or more different labels.')  #
    # assert (len(y_true) == len(probPos), 'Two input parameters must have the same size.')
    # y_pred = (probPos >= 0.5).astype(np.int32)
    # TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()

    # Recall@N (R@N): R@N is the ratio of the number of true applicable source projects in the top N of the recommendation
    # list to K
    TASPnumN = 0 # Number of true applicable source projects in top N
    for i in range(N):
        if appList[i] in recList[:N]:
            TASPnumN += 1

    RecallatN = TASPnumN / K
    # Precision@N (P@N): P@N is the ratio of true applicable source projects in the top N recommended source projects.
    PrecisionatN = TASPnumN / N

    FMeasureatN = 2 * PrecisionatN * RecallatN / (PrecisionatN + RecallatN)
    # FMeasureatN = 2/(1/PrecisionatN + 1/RecallatN)

    # MAP (Mean Average Precision) is the mean of the average precision (AP) of multiple new projects, which reflects
    # the recommendation performance of the model as a whole.
    AP = 0
    for i in range(0, N):
        TASPnumi = 0  #
        if appList[i] in recList[:i]:  # 需要调试
                TASPnumi += 1

        Precisionati = TASPnumi / (i+1)
        if i in appList[:N]:
            reli = 1  # the ith recommended project is applicable.
        else:
            reli = 0  # the ith recommended project is not attached to applicable source projects.
        APi = Precisionati * reli
        AP += APi
    AP = AP / K

    MAP = 0
    for i in range(M):
        MAP += AP
    MAP = MAP / M

    # MRR (Mean Reciprocal Ranking) is related to the location of any applicable source project first retrieved in the
    # recommendation list. MRR is the average of the RR (Reciprocal Ranking).
    MRR = 0
    for i in range(M):
        ranki = recList[0]  # ranki represents the rank of any applicable source project first recommended to the ith new project
        RRi = 1 / ranki
        MRR += RRi
    MRR = MRR / M
    measures = {'F-Measure@N': FMeasureatN, 'MAP': MAP, 'MRR': MRR}


    return measures


def CFPS_main(mode, clf_index, runtimes, learnerIndex, measureString):  # , K, measureString, clf_index
    """

    :param mode:
    :param clf_index:
    :param runtimes:
    :return:
    """

    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.4.csv', 'xerces-init.csv'],
            ['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv'],
            ['EQ.csv', 'JDT.csv', 'LC.csv', 'ML.csv', 'PDE.csv'],
            ['Apache.csv', 'Safe.csv', 'Zxing.csv']]


    # # example
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

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

                    HPs = []
                    X_train_num = 0
                    for file_tr in datasets[i]:
                        if file_tr != file_te:
                            # print('train_file:', file_tr)
                            Address_tr = datapath + file_tr
                            Samples_tr = NumericStringLabel2BinaryLabel(Address_tr)  # original train data, DataFrame
                            Samples_tr.columns = column_name.tolist()  # 批量更改列名
                            X_train_num += len(Samples_tr)
                            # /* step1: data preprocessing */
                            Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                                = Random_Stratified_Sample_fraction(Samples_tr, string='bug',
                                                                    r=r)  # random sample 90% negative samples and 90% positive samples
                            Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                            data_train = pd.DataFrame(Sample_tr)
                            data_train_unique = Drop_Duplicate_Samples(data_train)  # drop duplicate samples
                            data_train_unique = data_train_unique.values
                            X_train = data_train_unique[:, : -1]
                            y_train = data_train_unique[:, -1]

                            HP = np.c_[X_train, y_train]
                            HPs.append(HP)
                            NP = np.c_[X_test, y_test]
                            # X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore',
                            #                                                   X_test)  # data transformation
                            # # source = np.concatenate((X_train_zscore, y_train), axis=0)  # array垂直拼接
                            # HP = np.c_[X_train_zscore, y_train]
                            # HPs.append(HP)
                            # NP = np.c_[X_test_zscore, y_test]

                    # /* step2: data filtering */
                    # *******************BF*********************************
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))

                    if len(datasets[i]) > 12:
                        K = 10
                    elif len(datasets[i]) > 4 and len(datasets[i]) < 12:
                        K = 3
                    elif len(datasets[i]) <= 4:
                        K = 1
                    else:
                        pass
                    # CFPS(HPs, NP, clf_index, r, measureString, K)
                    ASPs = CFPS(HPs, NP, learnerIndex, r, measureString, K)
                    if len(ASPs) == 1:
                        source = ASPs[0]
                    else:
                        source = ASPs[0]
                        for l in range(1, len(ASPs)):
                              source = np.concatenate((source, ASPs[l]), axis=0)

                    X_train = source[:, :-1]
                    y_train = source[:, -1]
                    X_train_zscore, X_test_zscore = Standard_Features(X_train, 'zscore',
                                                                      X_test)  # data transformation
                    X_train_new = X_train_zscore
                    y_train_new = y_train

                    # /* step3: Model building and evaluation */
                    # Train model: classifier / model requires the label must beong to {0, 1}.
                    modelname_bf, model_bf = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_bf)
                    # print("modelname_bf:", modelname_bf)
                    measures_bf = Build_Evaluation_Classification_Model(model_bf, X_train_new, y_train_new,
                                                                        X_test_zscore, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time

                    measures_bf.update(
                        {'train_len_before': X_train_num, 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_bf,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_bf, index=[r])
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
    # HPs = [np.array([[1, 0, 1, 0],
    #                 [1, 1, 4, 5]]),
    #        np.array([[1, 2, 3, 5],
    #                 [3, 1, 5, 0]]),
    #        np.array([[1, 0, 2, 5],
    #                 [0, 20, 5, 1]]),
    #        np.array([[0, 3, 7, 4],
    #                 [1, 4, 5, 10]])]
    # NP = np.array([[0, 5, 10, 1],
    #                [1, 5, 3, 10]])
    # GetSimiScores(HPs, NP)

    mode = ['null', 'M2O_CPDP', 'eg_CFPS']

    for i in range(0, 3, 1):
        CFPS_main(mode=mode, clf_index=i, runtimes=2, learnerIndex=1, measureString='F1')







