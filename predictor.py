'''
@author: Unknown
@update: Zhouhong Gu
@data: 2021/10/13
'''

# coding=utf-8

import os
from sklearn import linear_model, ensemble
from sklearn.feature_selection import chi2
from sklearn import preprocessing
import numpy as np
import sample
from tools import LoadCSV
from config import datapath

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('ggplot')


def getOriginalData(path):
    all_data = LoadCSV(path)
    # dataY 是否应该被采样
    dataY = [ff[1] for ff in all_data]
    # dataX 一些参数，可以用来预测是否应该被采样
    dataX = [ff[3:] for ff in all_data]
    return dataX, dataY


def TestModel(model, k):
    global predY
    if k == 't':
        model.fit(trainX, trainY)
        ptrainY = model.predict(trainX)
        trainloss = ((ptrainY - trainY) ** 2).sum() / trainY.shape[0]
        predY = model.predict(testX)
        testloss = ((predY - testY) ** 2).sum() / testY.shape[0]
        print('trainloss=%.5f  testloss=%.5f' % (trainloss, testloss))
    else:
        model.fit(trainX_1, trainY)
        ptrainY = model.predict(trainX_1)
        trainloss = ((ptrainY - trainY) ** 2).sum() / trainY.shape[0]
        predY = model.predict(testX_1)
        testloss = ((predY - testY) ** 2).sum() / testY.shape[0]
        print('trainloss=%.5f  testloss=%.5f' % (trainloss, testloss))


def GetF1(strata=10):
    sssample = []
    thres = np.arange(0, 0.25, 0.01)
    for thre in thres:
        tp = (((predY > thre) * 1 + (testY >= 0.25) * 1) == 2).sum()
        pp = (testY > 0).sum()
        oo = (predY > thre).sum()
        prec = tp / oo
        reca = tp / pp
        f1 = 2 * prec * reca / (prec + reca + 10 ** (-5))

        qqq = sample.sampling([[x * 4, y * 4] for x, y in zip(predY, testY)], strata)
        sample_size = sum([len(item) for item in qqq.values()])
        sample_mean = sum([sum(qqq[i]) for i in range(strata)]) / sample_size
        print('sample mean', sample_mean)
        print('sample size', sample_size)
        sssample.append(sample_size)

        pgs = [(x, y) for x, y in zip(predY, testY)]
        pgs.sort()
    pgs = [(x, y) for x, y in zip(predY, testY)]
    pgs.sort()
    prcret = []
    alltp = len([x for x in pgs if x[1] >= 0.25])
    tp = alltp
    for k, pg in enumerate(pgs):
        pp, gg = pg
        prec = tp / (len(pgs) - k)
        reca = tp / alltp
        prcret.append((prec, reca))
        if gg >= 0.25: tp -= 1
    prcret = list(reversed(prcret))
    auc = 0
    for ii in range(1, len(prcret)):
        auc += (prcret[ii - 1][0] + prcret[ii][0]) * (prcret[ii][1] - prcret[ii - 1][1]) * 0.5
    print('auc = %.5f' % auc)
    return prcret, np.mean(sssample), np.std(sssample)


def MakeSingleDataset(dataX, dataY):
    X, Y = [], []
    for zx, zy in zip(dataX, dataY):
        Y.append(int(zy))
        xx = [eval(x) for x in zx]
        X.append(xx)
    X = np.array(X)
    Y = np.array(Y)
    np.random.seed(2333)
    np.random.shuffle(X)
    np.random.seed(2333)
    np.random.shuffle(Y)
    np.seterr(divide='ignore', invalid='ignore')
    X[:, 3:] = np.log(X[:, 3:] + 1)
    mms = preprocessing.MinMaxScaler()
    X = mms.fit_transform(X)
    X_1 = (X[:, 2:]).copy()
    Y = Y * 0.25
    return X, Y, mms, X_1


def getEntropy(D):
    """
    Calculate and return entropy of 1-dimensional numpy array D
    """
    length = len(D)
    valueList = list(set(D))
    numVals = len(valueList)
    countVals = np.zeros(numVals)
    Ent = 0
    for idx, val in enumerate(valueList):
        countVals[idx] = len([x for x in D if x == val])
        Ent += countVals[idx] * 1.0 / length * np.log2(length * 1.0 / countVals[idx])
    return Ent


def getMaxInfoGain(D, X, feat=0):
    """
    Calculate maximum information gain w.r.t. the feature which is specified in column feat of the 2-dimensional array X.
    """
    D = np.array(D)
    EntWithoutSplit = getEntropy(D)
    feature = X[:, feat]
    length = len(feature)
    valueList = list(set(feature))
    splits = np.diff(valueList) / 2.0 + valueList[:-1]
    maxGain = 0
    bestSplit = 0
    bestPart1 = []
    bestPart2 = []
    for split in splits:
        Part1idx = np.argwhere(feature <= split)
        Part2idx = np.argwhere(feature > split)
        E1 = getEntropy(D[Part1idx[:, 0]])
        l1 = len(Part1idx)
        E2 = getEntropy(D[Part2idx[:, 0]])
        l2 = len(Part2idx)
        Gain = EntWithoutSplit - (l1 * 1.0 / length * E1 + l2 * 1.0 / length * E2)
        if Gain > maxGain:
            maxGain = Gain
            bestSplit = split
            bestPart1 = Part1idx
            bestPart2 = Part2idx
    return maxGain, bestSplit, bestPart1, bestPart2


featurename = [str(i) for i in range(11)]


def GetInformationGain(X, y):
    E = getEntropy(y)
    print("Entropy of Class Labels= ", E)
    ret = []
    for col in range(X.shape[1]):
        print('-' * 30)
        print("Best split w.r.t. to feature %s" % featurename[col])
        maxG, bestSplit, Part1, Part2 = getMaxInfoGain(y, X, feat=col)
        print("Maximum Information Gain = ", maxG)
        print("Best Split = ", bestSplit)
        print("Samples in partition 1: ", len(Part1))
        print("Samples in partition 2: ", len(Part2))
        ret.append(maxG)
    return ret


def TryModels():
    global mrf, mridge, prcret, predY

    plt.figure()
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    plt.xlabel('Recall', fontsize=19)
    plt.ylabel('Precision', fontsize=19)

    print('ridge_entity')
    mridge = linear_model.RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0])
    TestModel(mridge, 'e')
    sssample1 = []
    sssample1_std = []
    for ss in range(2, 16):
        print("strata:", ss)
        prcret, aaa, var1 = GetF1(strata=ss)
        sssample1.append(aaa)
        sssample1_std.append(var1)
    print(sssample1)
    print(chi2(trainX_1, trainY >= 0.25))
    print(mridge.coef_)
    print('rf_entity')
    mrf = ensemble.RandomForestRegressor(50, verbose=False)
    TestModel(mrf, 'e')
    sssample2 = []
    sssample2_std = []
    for ss in range(2, 16):
        print("strata:", ss)
        prcret, aaa, var2 = GetF1(strata=ss)
        sssample2.append(aaa)
        sssample2_std.append(var2)
    print(sssample2)
    plt.plot([y for x, y in prcret], [x for x, y in prcret])

    lgd = plt.legend(['baseline', 'linear-entity', 'linear-triple', 'RF-entity', 'RF-triple'], loc=4, fontsize=18)
    lgd.get_frame().set_alpha(0.5)
    plt.savefig('predictort2.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 获取数据集
    path = os.path.join(datapath, 'main_ent_info.txt')
    dataX, dataY = getOriginalData(path)
    X, Y, mms, X_1 = MakeSingleDataset(dataX, dataY)

    # 数据集划分
    spos = int(X.shape[0] * 0.7)
    trainX, testX = X[:spos], X[spos:]
    trainX_1, testX_1 = X_1[:spos], X_1[spos:]
    trainY, testY = Y[:spos], Y[spos:]

    rep = 1
    mean_size1 = []
    mean_size2 = []
    for i in range(rep):
        TryModels()

    # 这个模型预测的是实体被修改的概率
    # 如果一个实体被修改的概率比较大，则这个实体被采样用来做实时性评估
    GetInformationGain(trainX[:20000], trainY[:20000])
    mrf = ensemble.RandomForestRegressor(100, verbose=False)
    mrf.fit(X, Y)
