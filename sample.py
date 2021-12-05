'''
@author: Kexin Qin
@update: Zhouhong Gu
@data: 2021/10/13
'''

import numpy as np
import pandas as pd
import collections

# 分层
def stratify(data, k):
    """
    实现 Minimum Variance Stratification (MVS) 分层算法
    :param data: 要分层的数据，可以放在一个list里面
    :param k: 层数
    :return: data_strata: 列表，eg.[0,1,1,2,2] 表示分三层，其中第一层(0)一个，第二层(1) 两个，第三层(2) 两个
    """
    length = 1 / k
    hist, bin_edges = np.histogram(data, bins=2000)
    H = np.cumsum(np.sqrt(hist))
    H = H / max(H)
    start = 0
    strata = []
    for i in range(k - 1):
        point = (i + 1) * length
        for j in range(start, len(H) - 1):
            if H[j] < point and H[j + 1] > point:
                start = j + 1
                if abs(H[j] - point) < abs(H[j + 1] - point):
                    strata.append((bin_edges[j] + bin_edges[j + 1]) / 2)
                else:
                    strata.append((bin_edges[j + 1] + bin_edges[j + 2]) / 2)
                break
    data_strata = []
    for ii in range(len(data)):
        for pp in range(len(strata)):
            if data[ii] < strata[pp]:
                data_strata.append(int(pp))
                break
            if data[ii] > strata[-1]:
                data_strata.append(int(len(strata)))
                break
    return data_strata

# 单层
def simple_stra(data, k):
    """
    实现“简单分层”，即按照quantile来分，前1/k为第一层，依次类推。和上面MVS的区别主要在于H的计算方式
    MVS:     H = np.cumsum(np.sqrt(hist))
    simple:  H = np.cumsum(hist)
    """
    length = 1 / k
    hist, bin_edges = np.histogram(data, bins=2000)
    H = np.cumsum(hist)
    H = H / max(H)
    start = 0
    strata = []
    for i in range(k - 1):
        point = (i + 1) * length
        for j in range(start, len(H) - 1):
            if H[j] < point and H[j + 1] > point:
                start = j + 1
                if abs(H[j] - point) < abs(H[j + 1] - point):
                    strata.append((bin_edges[j] + bin_edges[j + 1]) / 2)
                else:
                    strata.append((bin_edges[j + 1] + bin_edges[j + 2]) / 2)
                break
    data_strata = []
    for ii in range(len(data)):
        for pp in range(len(strata)):
            if data[ii] < strata[pp]:
                data_strata.append(int(pp))
                break
            if data[ii] > strata[-1]:
                data_strata.append(int(len(strata)))
                break
    return data_strata

def GMM(data, k):
    """
    GMM,高斯混合模型，类似于K-means，也是实现分层的一种方法（把分层当做聚类问题）
    """
    from sklearn.mixture import GaussianMixture
    clf = GaussianMixture(n_components=k)
    clf.fit([[data[i], 0] for i in range(len(data))])
    result = clf.predict([[data[i], 0] for i in range(len(data))])
    return result

# 采样
def sampling(population, k):
    """
    :param population: 即一个分布，在本案例中为预测值pred的分布
    :param k:
    :return: sample_data: 要到达给定的MOE条件(由epsilon和llevel决定)，从population中每一层采出的样本值
             eg.  {0: 0.5, 1: 0.2, 0.3} 意思是从第0层采样了一次，值为0.5， 从第1层采样两次，值分别为0.2,0.3
    """
    size1 = 40
    epsilon = 0.01
    llevel = 1.96 # 1.64, 1.96
    if not k:
        df = pd.DataFrame(population, columns=['value'])
        d1 = [item for item in df.sample(n=size1)['value']]
        while llevel*np.sqrt(np.mean(d1)*(1-np.mean(d1))/len(d1)) > epsilon:
            d2 = [item for item in df.sample(n=size1)['value']]
            d1.extend(d2)
        return d1
    else:
        df = pd.DataFrame(population, columns=['pred', 'test'])
        classes = stratify(df['pred'], k)
        df['classes'] = classes
        ds = df.groupby("classes")
        ggg = [item for item in classes]
        stra_len = [ggg.count(i) for i in range(k)]
        typicalFracDict = dict()
        typicalFracDict[0] = stra_len[0] / len(classes)
        for i in range(1, k):
            typicalFracDict[i] = stra_len[i]/len(classes)+typicalFracDict[i-1]

        sample_data = collections.defaultdict(list)

        def choose_group(random_num, Dict):
            """
            :param random_num:  0-1 范围内随机数
            :param Dict: 占比
            :return: 一个list，包含每一次应该从第几层采样，与random_num等长，根据随机数落在0-1中的位置来选择采样层，保证每一层概率上的一致
            """
            ret = []
            for item in random_num:
                for jj in range(len(Dict)):
                    if item<Dict[jj]:
                        ret.append(jj)
                        break
                    if item>Dict[len(Dict)-1]:
                        ret.append(len(Dict)-1)
                        break
            return ret

        kkk = np.random.random(100)
        groups = choose_group(kkk, typicalFracDict)
        for item in groups:
            sample_data[item].append(ds.get_group(item).sample(n=1).iloc[0, 1])
        while llevel*np.sqrt(stra_var(sample_data, stra_len)) > epsilon:
            kkk = np.random.random(size1)
            groups = choose_group(kkk, typicalFracDict)
            for item in groups:
                sample_data[item].append(ds.get_group(item).sample(n=1).iloc[0, 1])
        return sample_data

# 某一层中的方差
def stra_var(samples, stra_len):
    """
    :param samples: 样本
    :param stra_len: 分层中的总样本数
    :return: 该分层的方差
    """
    var = 0
    total = sum(stra_len)
    for ii in range(len(stra_len)):
        item = samples[ii]
        if len(item) <= 1:
            var += 1
        else:
            var += ((stra_len[ii]/total)**2)*(1-len(item)/stra_len[ii])*np.var(item, ddof=1)/len(item)
    return var


