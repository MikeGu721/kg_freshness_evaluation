'''
@author: Kexin Qin
@update: Zhouhong Gu
@data: 2021/10/13
'''

from datetime import datetime
from tools import LoadCSV
from config import datapath
import os



def period(strtime):
    '''
    返回strtime中的内容与T_now的差距
    @strtime: 查询时间
    '''
    return (T_now - datetime.strptime(strtime, '%Y-%m-%d')).total_seconds() // 86400


# 当前的时间
T_now = datetime.strptime('2020-07-24', '%Y-%m-%d')
# 获得实体
entity_data = LoadCSV(os.path.join(datapath, 'main_ent_info.txt'))
all_ent = [ff[0] for ff in entity_data]
# 获得三元组
triple_data = LoadCSV(os.path.join(datapath, 'main_triple_info.txt'))
# 获得property
predicate_data = LoadCSV(os.path.join(datapath, 'prop_sort.txt'))
predicate = dict()
for item in predicate_data:
    # property的某种指标
    predicate[item[0]] = eval(item[1])

ent_start = 0
ent_ind = 0
tri_out = []

# 输出<s,p,o>,p的指标,实体的指标
for tri in triple_data:
    (ent, p, _) = eval(tri[0])
    ent_ind = ent_start
    tri1 = tri.copy()
    while ent_ind < len(entity_data):
        if all_ent[ent_ind] == ent:
            ent_start = ent_ind
            if p in predicate:
                tri1.append(predicate[p])
            else:
                tri1.append(0.82)
            tri1.extend(entity_data[ent_ind][3:])
            tri_out.append(tri1)
            break
        else:
            ent_ind += 1

with open(os.path.join(datapath, 'main_tri_info.txt'), 'w', encoding='utf-8') as f1:
    for item in tri_out:
        f1.write('\t'.join([str(x) for x in item]) + '\n')
        f1.flush()

