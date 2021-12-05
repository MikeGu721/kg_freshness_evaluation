'''
@author: Zhouhong Gu
@data: 2021/10/20
'''
import os
import random

# put the dbpedia file in this dir
datapath = './dataset/'


def get_txtdata(name):
    return os.path.join(datapath, 'dbpedia_%s_txt' % name)


def get_triple_data(name, num=10000):
    f = open(os.path.join(datapath, '%s_unfresh_triple.txt' % name), encoding='utf-8')
    unfresh_triple = list(random.sample([i.strip() for i in f], k=num))
    fresh_triple = []
    for file in os.listdir(os.path.join(datapath, get_txtdata(name))):
        f = open(os.path.join(datapath, get_txtdata(name), file), encoding='utf-8')
        for index, line in enumerate(f):
            fresh_triple.append(line.strip())
            if index == num:
                break
    fresh_triple = random.sample(fresh_triple, k=num)
    return fresh_triple, unfresh_triple


def get_entity_data(name, num=10000):
    f = open(os.path.join(datapath, '%s_unfresh_entity.txt' % name), encoding='utf-8')
    unfresh_entity = list(random.sample([i.strip() for i in f], k=num))
    fresh_entity = []
    for file in os.listdir(os.path.join(datapath, get_txtdata(name))):
        f = open(os.path.join(datapath, get_txtdata(name), file), encoding='utf-8')
        for index, line in enumerate(f):
            fresh_entity.append(line.strip().split('\t')[0])
            if index == num:
                break
    fresh_entity = random.sample(fresh_entity, k=num)
    return fresh_entity, unfresh_entity
