# encoding=utf-8

"""
    Created on 21:29 2018/11/12
    @author: Jindong Wang

"""

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier


def kernel(ker, X1, X2, gamma):
    '''

    :param ker:
    :param X1:
    :param X2:
    :param gamma:
    :return:
    '''

    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(X1.T, X2.T)  # np.asarray(X1).T, np.asarray(X2).T
        else:
            K = sklearn.metrics.pairwise.linear_kernel(X1.T)  # np.asarray(X1)

    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(X1.T, X2.T, gamma)  # np.asarray(X1).T, np.asarray(X2).T
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(X1.T, None, gamma)  # np.asarray(X1).T
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: int, dimension after transfer, less than ns+nt
        :param lamb: non-negative value, lambda value in equation  # regularization parameter
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''

        ns, nt = len(Xs), len(Xt)
        X = np.hstack((Xs.T, Xt.T))  # 横向拼接n_feature x ns 与 n_feature x nt拼接， n_feature x (nt+ns)
        # print('shape1:', X.shape)
        X /= np.linalg.norm(X, axis=0)  # 归一化？ 2-范数，求整个矩阵元素平方和再开根号，axis=0表示按列向量处理，求多个列向量的范数
        n_feature, n = X.shape
        # print('shape2:', X.shape)

        # Kernel function K
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)  # (n x n = (nt+ns) x (nt+ns))
        # print('kernel:', K, K.shape)
        # L matrix: {Lij}, dim:(nt+ns) x (nt+ns)

        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))  # 按垂直方向叠加
        L = e * e.T
        # print(L)
        L = L / np.linalg.norm(L, 'fro')  # F-范数, 归一化？

        #  Center matrix  H
        m = n_feature if self.kernel_type == 'primal' else n
        Im = np.eye(m)
        one = np.ones((n, n))  # all 1 in I:  n = ns + nt
        H = Im - 1 / n * one * one.T  # eye - identity matrix ,对角线为1，其余为0

        # similar to kernel Fisher discriminant analysis, the W solutions
        Matrix = np.linalg.inv(K * L * K + self.lamb * Im) + K * H * K.T
        # tem = np.linalg.inv(np.linalg.multi_dot([K, L, K]) + self.lamb * Im)
        # Matrix = tem * np.linalg.multi_dot([K, H, K])

        # compute W matrix
        w, V = scipy.linalg.eig(Matrix)  # 矩阵A的全部特征值，构成对角阵w，并求A的特征向量构成V的列向量。
        # ind = np.argsort(w)[::-1].tolist()   # 特征向量降序排列
        ind = np.argsort(w)
        # m1 = n - 1  # Matrix的秩
        # W = V[:, ind[:m1]]  # m1 leading eigenvectors of Matrix = (kLK+uI)^(-1) KHK, m1<= ns+nt-1
        W = V[:, ind[:self.dim]]  # dim after transfer
        # print('W:', W.shape, type(W), type(K))

        # compute K~
        # K_x = W.T * K
        K_x = np.dot(W.T, K).astype(float)  # ComplexWarning: Casting complex values to real discards the imaginary part
        # print(K_x.shape)
        K_x /= np.linalg.norm(K_x, axis=0)  # 归一化？
        Xs_new, Xt_new = K_x[:, :ns].T, K_x[:, ns:].T
        print('after transformation:', Xs_new.shape, Xt_new.shape)
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)

        return acc, y_pred


if __name__ == '__main__':

    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']

    for i in [2]:
        for j in [3]:
            if i != j:
                src, tar = 'data/' + domains[i], 'data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
                acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt)

                print(acc)

                # It should print 0.910828025477707