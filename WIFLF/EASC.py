"""
Effort-Aware Supervised Cross-project defect prediction (EASC) belongs to strict M2O-CPDP.
EASC assumes that for these identified potential defective instances, the instances with higher defect-proneness should
be inspected first.
Technical
REFERENCE:
    Ni, Chao, Xin Xia, David Lo, Xiang Chen, and Qing Gu. "Revisiting supervised and unsupervised methods for effort-
    aware cross-project defect prediction." IEEE Transactions on Software Engineering (2020).

"""

from copy import deepcopy
import os
import time
import numpy as np
import Performance
import pandas as pd
from sklearn import metrics
from DataProcessing import Check_NANvalues, DataFrame2Array, DataImbalance, Delete_abnormal_samples, \
     DevideData2TwoClasses, Drop_Duplicate_Samples, indexofMinMore, indexofMinOne, NumericStringLabel2BinaryLabel, \
     Random_Stratified_Sample_fraction, Standard_Features, Log_transformation  # Process_Data,
from ClassificationModel import Selection_Classifications, Build_Evaluation_Classification_Model

def EASC(projects, classifier, r, effort=[0.2, 1000, 2000]):
    """
    FUNCTION EASC() SUMMARY GOES HERE:
    An improved supervised file-level CPDP method EASC.
    :param projects: list, all projects in a specific dataset, the type of each project is DataFrame;
    :param classifier: the basic classifier, e.g., Naive Bayes;
    :param effort: the available effort to decide whether a instance is defective or not, the default is 20% total lines;
    :return: Results: a list which contains all performance pairs of non-effort-aware measures and effort-aware measures
             (e.g., (NPM,EPM));
    """

    "===STEP1: model building phase=="
    # 1: Filter unsuitable projects from projects;
    projectsNum1 = len(projects)
    suitableProjects = deepcopy(projects)
    # projects will be removed if they do not have the required minimum number of instances (i.e., 5; following
    # the same setting as Herbold et al. [28]) in each class (i.e., defective and non-defective)
    for i in range(projectsNum1):
        uniqueList = np.unique(projects[i].values[:, -1]).tolist()
        if len(uniqueList) == 1:  # or uniqueList[1]==1:
            suitableProjects.remove(projects[i])  # remove the project which only has one class (defective or non-defective)
        elif (len(uniqueList) == 2) and (projects[i].values[:, -1].tolist().count(uniqueList[0]) < 5 or projects[i].values[:, -1].tolist().count(uniqueList[1]) < 5):
            suitableProjects.remove(projects[i])  # remove  the project which has less 5 instances in each class
        else:
            pass
    projectsNum2 = len(suitableProjects)
    if projectsNum2 <= 1:
        print("The projects are not enough for CPDP. Please check!")

    # 2: for all TestProject in projects do
    EPMResults = pd.DataFrame()
    NPMResults = pd.DataFrame()
    for i in range(projectsNum2):
        TestProject = suitableProjects[i]  # DataFrame
        # 3: TrainSet = Set(a copy of projects)-Set(TestProject, any other versions of TestProject);
        TrainSet = deepcopy(suitableProjects)  # DataFrame
        del TrainSet[i]
        # TrainSet.remove(suitableProjects[i])  # TrainingProjects: list includes DataFrame
        # 4: Build a predictor by using classiffer on TrainSet;

        TrainData = TrainSet[0].values
        for k in range(1, len(TrainSet)):
            TrainXy = np.concatenate((TrainData, TrainSet[k].values), axis=0)  # array垂直拼接
        X_train, y_train = TrainXy[:, :-1], TrainXy[:, -1]

        # TrainData = pd.DataFrame()
        # for k in range(len(TrainSet)):
        #     TrainData = pd.concat([TrainData, TrainSet[k]])
        # X_train, y_train = TrainData.values[:, :-1], TrainData.values[:, -1]

        classifier.fit(X_train, y_train)
        testTrueLabels = TestProject.values[:, -1]
        TestPredictProb = classifier.predict_proba(TestProject.values[:, :-1])
        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        # print(X_train, X_train.shape, y_train, y_train.shape,
        #       X_test, X_test.shape, y_test, y_test.shape, classifier.classes_)
        for j in range(TestPredictProb.shape[0]):  # 每个类的概率, array类型
            if len(classifier.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if classifier.classes_[1] == 1:
                    prob_pos.append(TestPredictProb[j][1])
                    prob_neg.append(TestPredictProb[j][0])
                else:
                    prob_pos.append(TestPredictProb[j][0])
                    prob_neg.append(TestPredictProb[j][1])
        prob_pos = np.array(prob_pos)

        "===STEP2: model evaluating phase==="
        EffortMetrics = ["LOC EXECUTABLE", "loc", "numberoflinesofcode", "countlinecode"] # NASA, PROMISE, AEEEM, RELINK
        LOCvector = []
        metricsName = TestProject.columns.values
        for effortMetric in metricsName:
            if effortMetric in EffortMetrics:
                LOCName = effortMetric
                LOCvector = TestProject[LOCName].values.tolist()
        LOCvector = np.array(LOCvector)
        # 5: (NPM;EPM)=EASC:ModelEvaluating(classifier,TestProject,effort), and append them to Results;
        EPMmeasures = EPM(LOCvector, testTrueLabels, prob_pos, effort)  # self-defined func, Calculating EPMs: the performance value of effort-aware performance measures
        NPMmeasures = NPM(LOCvector, testTrueLabels, prob_pos)  # self-defined func, Calculating NPMs: the performance value of non-effort-aware performance measures
        print("EPM:", EPMmeasures, "\n NPM:", NPMmeasures)
        EPMResults = pd.concat([EPMResults, pd.DataFrame(EPMmeasures, index=[r])], axis=0, sort=False,
                               ignore_index=False)  # the measures of all files in runtimes
        NPMResults = pd.concat([NPMResults, pd.DataFrame(NPMmeasures, index=[r])], axis=0, sort=False,
                               ignore_index=False)  # the measures of all files in runtimes



    return EPMResults, NPMResults


def main(mode, clf_index, r):
    pwd = os.getcwd()
    print(pwd)
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # print(father_path)
    datapath = father_path + '/dataset-inOne/'
    spath = father_path + '/results/'
    if not os.path.exists(spath):
        os.mkdir(spath)

    preprocess_mode = mode[0]
    train_mode = mode[1]
    save_file_name = mode[2]
    df_file_measures = pd.DataFrame()  # the measures of all files in all runtimes
    classifiername = []

    datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv', 'jedit-3.2.csv', 'log4j-1.0.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.0.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.4.csv', 'xerces-init.csv'],
            ['ant-1.7.csv', 'arc-1.csv', 'camel-1.6.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'log4j-1.1.csv', 'lucene-2.0.csv', 'poi-2.0.csv', 'redaktor-1.csv', 'synapse-1.2.csv', 'tomcat-6.0.389418.csv', 'velocity-1.6.csv', 'xalan-2.6.csv', 'xerces-1.3.csv'],
            ['EQ.csv', 'JDT.csv', 'LC.csv', 'ML.csv', 'PDE.csv'],
            ['Apache.csv', 'Safe.csv', 'Zxing.csv']]


    # example
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]
    # datasets = [['ant-1.3.csv', 'arc-1.csv', 'camel-1.0.csv', 'ivy-1.4.csv'], ['Apache.csv', 'Safe.csv', 'Zxing.csv']]

    datanum = 0
    for i in range(len(datasets)):
        datanum = datanum + len(datasets[i])

    n = 0
    EPMResultsAll = pd.DataFrame()
    NPMResultsAll = pd.DataFrame()
    for i in range(len(datasets)):
        projects = []
        for file in datasets[i]:
            n = n + 1
            print('----------%s:%d/%d------' % ('Dataset', n, datanum))
            # print('testfile', file)
            start_time = time.time()

            Address = datapath + file
            project = NumericStringLabel2BinaryLabel(Address)  # DataFrame
            projects.append(project)

        modelname, classifier = Selection_Classifications(clf_index, r)  # select classifier
        classifiername.append(modelname)
        EPMResults, NPMResults = EASC(projects, classifier, r=n, effort=[0.2, 1000, 2000])

        end_time = time.time()

        EPMResultsAll = pd.concat([EPMResultsAll, EPMResults], ignore_index=False, axis=0, sort=False)
        NPMResultsAll = pd.concat([NPMResultsAll, NPMResults], ignore_index=False, axis=0, sort=False)

    modelname = np.unique(classifiername)

    pathname_NPM = spath + '\\' + (save_file_name + '_NPM_' + modelname[0] + '.csv')
    NPMResultsAll.to_csv(pathname_NPM)

    pathname_EPM = spath + '\\' + (save_file_name + '_EPM_' + modelname[0] + '.csv')
    EPMResultsAll.to_csv(pathname_EPM)

    return EPMResultsAll, NPMResultsAll

def EASC_main(mode, clf_index, runtimes):

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


    # example
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
                    Samples_tr_all = pd.DataFrame()  # initialize the candidate training data of all cp data
                    for file_tr in datasets[i]:
                        if file_tr != file_te:
                            # print('train_file:', file_tr)
                            Address_tr = datapath + file_tr
                            Samples_tr = NumericStringLabel2BinaryLabel(Address_tr)  # original train data, DataFrame
                            Samples_tr.columns = column_name.tolist()  # 批量更改列名
                            # filter the unsuitable projects for EASC
                            uniqueList = np.unique(np.array(Samples_tr)[:, -1]).tolist()
                            if len(uniqueList) == 1:  # or uniqueList[1]==1:
                                pass  # remove the project which only has one class (defective or non-defective)
                            elif (len(uniqueList) == 2) and (
                                    np.array(Samples_tr)[:, -1].tolist().count(uniqueList[0]) < 5 or
                                    np.array(Samples_tr)[:, -1].tolist().count(uniqueList[1]) < 5):
                                pass  # remove  the project which has less 5 instances in each class
                            else:
                                Samples_tr_all = pd.concat([Samples_tr_all, Samples_tr], ignore_index=False, axis=0,
                                                       sort=False)

                    # /* step1: data preprocessing */
                    Sample_tr_pos, Sample_tr_neg, Sample_pos_index, Sample_neg_index \
                        = Random_Stratified_Sample_fraction(Samples_tr_all, string='bug',
                                                            r=r)  # random sample 90% negative samples and 90% positive samples
                    Sample_tr = np.concatenate((Sample_tr_neg, Sample_tr_pos), axis=0)  # array垂直拼接
                    # data_train = pd.DataFrame(Sample_tr)
                    X_train_new, y_train_new = Sample_tr[:, :-1], Sample_tr[:, -1]

                    # /* step3: Model building and evaluation */
                    method_name = mode[1] + '_' + mode[2]  # scenario + filter method
                    print('----------%s:%d/%d------' % (method_name, r + 1, runtimes))
                    # Train model: classifier / model requires the label must beong to {0, 1}.
                    modelname_bf, model_bf = Selection_Classifications(clf_index, r)  # select classifier
                    classifiername.append(modelname_bf)
                    # print("modelname_bf:", modelname_bf)
                    measures_bf = Build_Evaluation_Classification_Model(model_bf, X_train_new, y_train_new,
                                                                        X_test, y_test)  # build and evaluate models
                    end_time = time.time()
                    run_time = end_time - start_time

                    measures_bf.update(
                        {'train_len_before': len(Sample_tr), 'train_len_after': len(X_train_new), 'test_len': len(X_test),
                         'runtime': run_time, 'clfindex': clf_index, 'clfname': modelname_bf,
                         'testfile': file_te, 'trainfile': 'More1', 'runtimes': r + 1})
                    df_m2ocp_measures = pd.DataFrame(measures_bf, index=[r])
                    # print('df_m2ocp_measures:\n', df_m2ocp_measures)
                    df_r_measures = pd.concat([df_r_measures, df_m2ocp_measures], axis=0, sort=False,
                                              ignore_index=False)
                else:
                    pass

            # print('df_file_measures:\n', df_file_measures)
            # print('所有文件运行一次的结果为：\n', df_file_measures)
            df_file_measures = pd.concat([df_file_measures, df_r_measures], axis=0, sort=False,
                                         ignore_index=False)  # the measures of all files in runtimes
            # df_r_measures['testfile'] = file_list
            # print('df_r_measures1:\n', df_r_measures)

    modelname = np.unique(classifiername)
    # pathname = spath + '\\' + (save_file_name + '_clf' + str(clf_index) + '.csv')
    pathname = spath + '\\' + (save_file_name + '_' + modelname[0] + '.csv')
    df_file_measures.to_csv(pathname)
    # print('df_file_measures:\n', df_file_measures)
    return df_file_measures



def EPM(LOCVector, y_true, prob_pos, effort=[0.2, 1000, 2000]):
    """
    Calculating EPMs: the performance value of effort-aware performance measures
    1) Sort separately instances in Defective and NonDefective in descending order by score/LOC (score means the probability
     of defect-prone outputted by prediction model; LOC means the lines of code of the instance);
    2) Append NonDefective to the end of Defective;
    3) Select those instances in front of Defective into TargetList and make sure that the total cost of them accounts
       for effort;
    4) Calculate effort-aware performance based on TargetList, Defective and classifier, then save them into EPM;
    :param LOCVector: A 1-D array of lines of code of all the instance, len=M
    :param y_true: A 1-D array that denotes test instances' actual labels, {0,1} where 1 is defective, 0 non-defective.
    :param prob_pos: A 1-D array, which denotes the probability of predicted being defective for each instance.
    :param effort: list, inspection proportion or inspection number of LOC in the list.
    :return: EPMmeasures, dict.
    """
    assert (np.unique(y_true) > 1, 'Please ensure that ''y_true'' includes two or more different labels.')  #
    assert (len(y_true) == len(prob_pos) == len(LOCVector), 'Three input parameters must have the same size.')

    k = IFA(LOCVector, y_true, prob_pos)

    PIIAT02 = PIIatL(LOCVector, y_true, prob_pos, L=effort[0])
    PIIAT1000 = PIIatL(LOCVector, y_true, prob_pos, L=effort[1])
    PIIAT2000 = PIIatL(LOCVector, y_true, prob_pos, L=effort[2])

    CostEffortAT02 = CostEffortatL(LOCVector, y_true, prob_pos, L=effort[0])
    CostEffortAT1000 = CostEffortatL(LOCVector, y_true, prob_pos, L=effort[1])
    CostEffortAT2000 = CostEffortatL(LOCVector, y_true, prob_pos, L=effort[2])

    EPMmeasures = {"IFA": k, "PII@20%": PIIAT02, "PII@1000": PIIAT1000, "PII@2000": PIIAT2000,
                   "CostEffort@20%": CostEffortAT02, "CostEffortAT1000": CostEffortAT1000, "CostEffortAT2000": CostEffortAT2000,
                   "Popt": Popt}

    return EPMmeasures

def NPM(LOCVector, y_true, prob_pos):
    """
     = = = Calculating NPMs: the performance value of non-effort-aware performance measures = = =
    Sort Defective in descending order by score*LOC, then calculate non-effort-aware and save them into NPM ;
    F1-score, AUC and PF
    :param LOCVector: A 1-D array of lines of code of all the instance, len=M
    :param y_true: A 1-D array that denotes test instances' actual labels, {0,1} where 1 is defective, 0 non-defective.
    :param prob_pos: A 1-D array, which denotes the probability of predicted being defective for each instance.
    :return: NPMmeasures, dict.
    """
    assert (np.unique(y_true) > 1, 'Please ensure that ''y_true'' includes two or more different labels.')  #
    assert (len(y_true) == len(prob_pos) == len(LOCVector), 'Three input parameters must have the same size.')

    originalOrder, SortedInstancesOrder = SortOrderbyScoreMultipleLOC(LOCVector, TestPredictProb=prob_pos)
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    y_pred = (prob_pos >= 0.5).astype(np.int32)
    for i in SortedInstancesOrder:
        if y_pred[i] == y_true[i] == 1:
            TP += 1
        elif y_pred[i] == y_true[i] == 0:
            TN += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            FN += 1
        else:
            FP += 1
    print("TN, TP, FP, FN:", TN, TP, FP, FN)

    try:
        F1 = metrics.f1_score(y_true, y_pred)
        AUC = metrics.roc_auc_score(y_true, prob_pos)
        PF = FP / (FP + TN)

        PD = metrics.recall_score(y_true, y_pred)
        MCC = metrics.matthews_corrcoef(y_true, y_pred)
        Fbeta = metrics.fbeta_score(y_true, y_pred, beta=2)
        G_Mean2 = np.sqrt(PD * (1 - PF))  # G_Mean2
        Pre = metrics.precision_score(y_true, y_pred)

        G_Measure = 2 * PD * (1 - PF) / (PD + (1 - PF))
        alpha = 4
        skewed_F_measure = ((1 + alpha) * Pre * PD) / (alpha * Pre + PD)

    except (IOError, ZeroDivisionError):
        F1 = 0
        AUC = 0
        PF = 0

        PD = 0
        MCC = 0
        Fbeta = 0
        G_Mean2 = 0
        Pre = 0

        G_Measure = 0
        skewed_F_measure = 0

    else:
        F1 = metrics.f1_score(y_true, y_pred)
        AUC = metrics.roc_auc_score(y_true, prob_pos)
        PF = FP / (FP + TN)

        PD = metrics.recall_score(y_true, y_pred)
        MCC = metrics.matthews_corrcoef(y_true, y_pred)
        Fbeta = metrics.fbeta_score(y_true, y_pred, beta=2)
        G_Mean2 = np.sqrt(PD * (1 - PF))  # G_Mean2
        Pre = metrics.precision_score(y_true, y_pred)

        G_Measure = 2 * PD * (1 - PF) / (PD + (1 - PF))
        alpha = 4
        skewed_F_measure = ((1 + alpha) * Pre * PD) / (alpha * Pre + PD)


    NPMmeasures = {"F1": F1, "AUC": AUC, "PF": PF,  # the authors used three measures in the paper
                   "PD": PD, "MCC": MCC, "Fbeta": Fbeta, "G_Mean2": G_Mean2, "Precision": Pre,
                   "skewed-F1": skewed_F_measure, "GMeasure": G_Measure}

    return NPMmeasures


def SortOrderbyScoreMultipleLOC(LOCVector, TestPredictProb):
    """
    FUNCTION SortOrderbyScoreMultipleLOC() SUMMARY GOES HERE:
    Sorting strategy of non-effort-aware performance measures(NPM) for EASC method:
     1): predicts testing instances, and divides all instances into two groups: Group Defective and Group Clean.
     2): all the instances in the two groups will be sorted by score*LOC, respectively.
     3): The two groups are combined together. In particular, these instances in Group Clean will be appended at the end
      of Group Defective.
    :param LOCVector: A 1-D array of lines of code of all the instance, len=M.
    :param TestPredictProb: A 1-D array, which denotes the probability of predicted being defective for each instance.
    :return:originalOrder: the original order of the instances from 0 to M-1.
            SortedInstancesOrder: the sorted order according to the sorting strategy of the EASC method (len=M).
    """
    originalOrder = []
    DefectiveIndex = []
    NonDefectiveIndex = []
    DefectiveScoreofLOC = []  # SCORE/LOC in Defective
    NonDefectiveScoreofLOC = []  # SCORE/LOC in NonDefective

    # STEP1: predicts testing instances, and divides all instances into two groups: Group Defective and Group Clean.
    for i in range(len(TestPredictProb)):
        originalOrder.append(i)
        # For instances in Group Defective, the probability of defect-proneness of each instance is larger than 0.5
        if TestPredictProb[i] >= 0.5:
            DefectiveIndex.append(i)
            DefectiveScoreofLOC.append(TestPredictProb[i] * LOCVector[i])
        # For instances in Group clean have less than 0.5 probability of defect-proneness.
        else:
            NonDefectiveIndex.append(i)
            NonDefectiveScoreofLOC.append(TestPredictProb[i] * LOCVector[i])

    # STEP2: all the instances in the two groups will be sorted by score * LOC, respectively.
    DefectiveIndexArray = np.argsort(-np.array(DefectiveScoreofLOC))  # rank from max to min, return the index of DefectiveScoreofLOC

    DefectiveOrder = []
    for i in DefectiveIndexArray.tolist():
        DefectiveOrder.append(DefectiveIndex[i])

    NonDefectiveIndexArray = np.argsort(-np.array(NonDefectiveScoreofLOC))
    NonDefectiveOrder = []
    for i in NonDefectiveIndexArray.tolist():
        NonDefectiveOrder.append(NonDefectiveIndex[i])

    # STEP3: The two groups are combined together. In particular, these instances in Group Clean will be appended at
    # the end of Group Defective.
    SortedInstancesOrder = DefectiveOrder + NonDefectiveOrder

    return originalOrder, SortedInstancesOrder

def SortOrderbyScoreDivideLOC(LOCVector, TestPredictProb):
    """
    FUNCTION SortOrderbyScoreDivideLOC() SUMMARY GOES HERE:
    Sorting strategy of effort-aware performance measures(EPM) for EASC method:
     1): predicts testing instances, and divides all instances into two groups: Group Defective and Group Clean.
     2): all the instances in the two groups will be sorted by score/LOC, respectively.
     3): The two groups are combined together. In particular, these instances in Group Clean will be appended at the end
      of Group Defective.
    :param LOCVector: A 1-D array of lines of code of all the instance, len=M.
    :param TestPredictProb: A 1-D array, which denotes the probability of predicted being defective for each instance.
    :return:originalOrder: the original order of the instances from 0 to M-1.
            SortedInstancesOrder: the sorted order according to the sorting strategy of the EASC method (len=M).
    """
    originalOrder = []
    DefectiveIndex = []
    NonDefectiveIndex = []
    DefectiveScoreofLOC = []  # SCORE/LOC in Defective
    NonDefectiveScoreofLOC = []  # SCORE/LOC in NonDefective

    # STEP1: predicts testing instances, and divides all instances into two groups: Group Defective and Group Clean.
    for i in range(len(TestPredictProb)):
        originalOrder.append(i)
        # For instances in Group Defective, the probability of defect-proneness of each instance is larger than 0.5
        if TestPredictProb[i] >= 0.5:
            DefectiveIndex.append(i)
            DefectiveScoreofLOC.append(TestPredictProb[i]/LOCVector[i])
        # For instances in Group clean have less than 0.5 probability of defect-proneness.
        else:
            NonDefectiveIndex.append(i)
            NonDefectiveScoreofLOC.append(TestPredictProb[i] / LOCVector[i])

    # STEP2: all the instances in the two groups will be sorted by score/LOC, respectively.
    DefectiveIndexArray = np.argsort(-np.array(DefectiveScoreofLOC))  # rank from max to min, return the index of DefectiveScoreofLOC

    DefectiveOrder = []
    for i in DefectiveIndexArray.tolist():
        DefectiveOrder.append(DefectiveIndex[i])

    NonDefectiveIndexArray = np.argsort(-np.array(NonDefectiveScoreofLOC))
    NonDefectiveOrder = []
    for i in NonDefectiveIndexArray.tolist():
        NonDefectiveOrder.append(NonDefectiveIndex[i])

    # STEP3: The two groups are combined together. In particular, these instances in Group Clean will be appended at
    # the end of Group Defective.
    SortedInstancesOrder = DefectiveOrder + NonDefectiveOrder

    return originalOrder, SortedInstancesOrder

# Suppose we have a dataset with M instances and N defective instances in total. After inspecting L lines of code,
# suppose we inspected m instances and observed n defective instances. Additionally, let’s consider that we inspected k
# instances when we find the first defective instance.

def IFA(LOCVector, y_true, prob_pos):
    """
    FUNCTION IFA() SUMMARY GOES HERE:
    The number of Initial False Alarms encountered before we find the first defective instance. It is computed as:
    IFA = k.
    Notice: The smaller of these measures’ value, the better of these methods’ performance.
    :param LOCVector: A 1-D array of lines of code of all the instance, len=N
    :param y_true: A 1-D array that denotes test instances' actual labels, {0,1} where 1 is defective, 0 non-defective.
    :param prob_pos: A 1-D array, which denotes the probability of predicted being defective for each instance.
    :return: k, int, IFA.
    """

    originalOrder, SortedInstancesOrder = SortOrderbyScoreDivideLOC(LOCVector, prob_pos)

    k = 0
    for i in SortedInstancesOrder:
        if (prob_pos[i] < 0.5 or y_true[i] == 0):  # -A OR -B == - (A and B)
            k += 1
        else:
            break
    return k



def PIIatL(LOCVector, y_true, prob_pos, L):
    """
    Function PIIatL() summary goes here:
    Proportion of Instances Inspected (m) when L LOC of all instances (M) are inspected. A high PII@L indicates that, under the
    same number of LOC to be inspected, developers need to inspect more instances.
    relative LOC of PII and absolute LOC of PII:
        # PII@20% = m / M ; where L accounts for 20% of total LOC
        # PII @1000 =m / M ; where L equals to 1000 LOC
        # PII @2000 =m / M ; where L equals to 2000 LOC
    Notice: The smaller of these measures’ value, the better of these methods’ performance.
    :param LOCVector: A 1-D array of lines of code of all the instance, len=M
    :param y_true: A 1-D array that denotes test instances' actual labels, {0,1} where 1 is defective, 0 non-defective.
    :param prob_pos: A 1-D array, which denotes the probability of predicted being defective for each instance.
    :param L: float, inspection proportion or inspection number of LOC.
    :return: PIIatL, float,  PII@L.
    :return:
    """

    originalOrder, SortedInstancesOrder = SortOrderbyScoreDivideLOC(LOCVector, prob_pos)
    sumLOCs = int(np.sum(LOCVector))

    TargetList = []
    # PII@20%
    if L > 0 and L < 1:
        tempSumLOCs = 0
        k = -1
        for i in SortedInstancesOrder:
            k += 1
            tempSumLOCs += LOCVector[i]
            # ratio1 = tempSumLOCs / sumLOCs
            # ratio2 = (tempSumLOCs + LOCVector[SortedInstancesOrder[k + 1]]) / sumLOCs
            # if ratio1 <= L and ratio2 > L:
            #     TargetList = SortedInstancesOrder[:k+1]  # First 20% LOCs
            #     break

            ratio = tempSumLOCs / sumLOCs
            # ratio2 = (tempSumLOCs + LOCVector[SortedInstancesOrder[k + 1]]) / sumLOCs
            if ratio > L:
                TargetList = SortedInstancesOrder[:k]  # First 20% LOCs
                break
            else:
                pass

    # PII@1000 or # PII@2000
    elif L > 1 and L <= sumLOCs:
        tempSumLOCs = 0
        k = -1
        for i in SortedInstancesOrder:
            tempSumLOCs += LOCVector[i]
            k += 1
            # if tempSumLOCs <= L and (tempSumLOCs + LOCVector[SortedInstancesOrder[k+1]]) > L:
            #     TargetList = SortedInstancesOrder[:k+1]
            if tempSumLOCs > L:
                TargetList = SortedInstancesOrder[:k]
                break
            else:
                pass
    elif L > sumLOCs or L==1:
        TargetList = SortedInstancesOrder

    M = len(y_true)
    m = len(TargetList)
    PIIatL = m / M

    return PIIatL



def CostEffortatL(LOCVector, y_true, prob_pos, L):
    """
    Function CostEffortatL() summary goes here:
     Proportion of inspected defective instances (n) among all the actual defective instances (N) when L LOC of all instances are
     inspected. The high CostEffort@L indicates more defective instances could be detected.
     relative LOC of CostEffort and absolute LOC of CostEffort:
      # CostEffort@20% = n / N ; where L accounts for 20% of total LOC
      # CostEffort@1000 =  n / N ; where L equals to 1000 LOC
      # CostEffort@2000 = n / N ; where L equals to 2000 LOC
    Notice: The larger of these measures’ value, the better of these methods’ performance.
    :param LOCVector: A 1-D array of lines of code of all the instance, len=M
    :param y_true: A 1-D array that denotes test instances' actual labels, {0,1} where 1 is defective, 0 non-defective.
    :param prob_pos: A 1-D array, which denotes the probability of predicted being defective for each instance.
    :param L: float, inspection proportion or inspection number of LOC
    :return: CostEffortatL, float,  CostEffort@L.
    """

    originalOrder, SortedInstancesOrder = SortOrderbyScoreDivideLOC(LOCVector, prob_pos)
    sumLOCs = int(np.sum(LOCVector))
    TargetList = []

    # PII@20%
    if L > 0 and L < 1:
        tempSumLOCs = 0
        k = -1
        for i in SortedInstancesOrder:
            k += 1
            tempSumLOCs += LOCVector[i]
            # ratio1 = tempSumLOCs / sumLOCs
            # ratio2 = (tempSumLOCs + LOCVector[SortedInstancesOrder[k+1]]) / sumLOCs
            # if ratio1 <= L and ratio2 > L:
            #     TargetList = SortedInstancesOrder[:k+1]  # First 20% LOCs
            #     break

            if tempSumLOCs / sumLOCs > L:
                TargetList = SortedInstancesOrder[:k]  # First 20% LOCs
                break

    # CostEffort@1000 or # CostEffort@2000
    elif L > 1 and L <= sumLOCs:
        tempSumLOCs = 0
        k = -1
        for i in SortedInstancesOrder:
            tempSumLOCs += LOCVector[i]
            k += 1
            # if tempSumLOCs <= L and (tempSumLOCs + LOCVector[SortedInstancesOrder[k+1]]) > L:
            #     TargetList = SortedInstancesOrder[:k+1]
            #     break
            if tempSumLOCs > L:
                TargetList = SortedInstancesOrder[:k]
                break
    elif L > sumLOCs or L==1:
        TargetList = SortedInstancesOrder
    print("%s Inspection Effort is %s" % (L, TargetList))

    n = 0
    for i in range(len(TargetList)):
        if prob_pos[int(TargetList[i])] >= 0.5 and y_true[int(TargetList[i])] == 1:
            n += 1
    N = np.sum(y_true)

    CostEffortatL = n / N

    return CostEffortatL

def Popt():
    """
    Popt is the normalized version of the effort-aware performance measure originally introduced by Mende and Koschke [54].
    The Popt is based on the concept of the “codechurn-based” Alberg diagram [55].
    Notice that the larger of these measures’ value, the better of these methods’ performance.

    :return:
    """

    return Popt

def demoTest():
    LOCVector = np.array([50, 180, 100, 500, 758, 1000, 80, 210])
    TestPredictProb = np.array([0.22, 0.97, 0.12, 0.33, 0.89, 0.12, 0.78, 0.48])
    # TestPredictProb = np.array([0.43, 0.86, 0.495, 0.45, 0.34, 0.33, 0.78, 0.32])
    ActualY = np.array([0, 1, 0, 0, 1, 0, 1, 0])

    # test func SortOrderbyScoreDivideLOC()
    originalOrder, SortedInstancesOrder = SortOrderbyScoreDivideLOC(LOCVector, TestPredictProb)
    print("order of EPMs:", originalOrder, SortedInstancesOrder)  # [0, 1, 2, 3, 4, 5, 6, 7] [6, 1, 4, 0, 7, 2, 3, 5]

    # test func IFA(), PIIatL(), CostEffortATL()
    for l in [0.2, 1000, 2000, 2900]:
        # 0.2 Inspection Effort is [6, 1]
        # IFA: 0 , PII@20%: 0.25 , CostEffort@20%: 0.6666666666666666
        # 1000 Inspection Effort is [6, 1]
        # IFA: 0 , PII@20%: 0.25 , CostEffort@20%: 0.6666666666666666
        # 2000 Inspection Effort is [6, 1, 4, 0, 7, 2, 3]
        # IFA: 0 , PII@20%: 0.875 , CostEffort@20%: 1.0
        # 2900 Inspection Effort is [6, 1, 4, 0, 7, 2, 3, 5]
        # IFA: 0 , PII@20%: 1.0 , CostEffort@20%: 1.0
        ifa = IFA(LOCVector, y_true=ActualY, prob_pos=TestPredictProb)
        PIIATL = PIIatL(LOCVector, y_true=ActualY, prob_pos=TestPredictProb, L=l)
        CostEffortATL = CostEffortatL(LOCVector, y_true=ActualY, prob_pos=TestPredictProb, L=l)
        print("L is", l, ", IFA:", ifa, ", PII@L:", PIIATL, ", CostEffort@L:", CostEffortATL)

    # test func EPM()
    # epmMeasures = EPM(LOCVector, y_true=ActualY, prob_pos=TestPredictProb, effort=[0.2, 1000, 2000])
    # print("EPMs:", epmMeasures)

    # test func SortOrderbyScoreMultipleLOC()
    originalOrder1, SortedInstancesOrder1 = SortOrderbyScoreMultipleLOC(LOCVector, TestPredictProb)
    print("order of NPMs:", originalOrder1, SortedInstancesOrder1)  # [0, 1, 2, 3, 4, 5, 6, 7] [4, 1, 6, 3, 5, 7, 2, 0]

    # test func EPM()
    npmMeasures = NPM(LOCVector, y_true=ActualY, prob_pos=TestPredictProb)
    print("NPMs:", npmMeasures)

if __name__ == '__main__':

    # y = [0, 0, 0, 0]
    # y = [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    # y = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # projects = [0, 1, 2, 3]
    # suitableProjects = deepcopy(projects)
    # for i in range(len(projects)):
    #     uniqueList = np.unique(y).tolist()
    #     if len(uniqueList) == 1:  # or uniqueList[1]==1:
    #         suitableProjects.remove(projects[i])
    #     elif (len(uniqueList) == 2) and (y.count(uniqueList[0]) < 5 or y.count(uniqueList[1]) < 5):
    #         suitableProjects.remove(projects[i])
    #     else:
    #         pass

    # demoTest()
    # EASC(projects, classifier, effort)

    mode = ['', 'M2O_CPDP', 'eg_EASC']
    # foldername = mode[1] + '_' + mode[2]
    # for i in range(0, 3, 1):
    #     EASC_main(mode=mode, clf_index=i, runtimes=2)

    main(mode=mode, clf_index=0, r=2)