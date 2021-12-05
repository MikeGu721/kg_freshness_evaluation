'''
@author: Zhouhong Gu
@data: 2021/10/20
'''
import config
import time
import os

txt = ['0601', '0901', 'live']
datapath = './dataset'
choosen = [0, 1]
batch_size = 10000

txt = [txt[i] for i in choosen]
unfresh_triple = '%s_unfresh_triple.txt' % txt[0]
unfresh_entity = '%s_unfresh_entity.txt' % txt[0]
unfresh_triple = os.path.join(datapath, unfresh_triple)
unfresh_entity = os.path.join(datapath, unfresh_entity)

dir1 = config.get_txtdata(txt[0])
dir2 = config.get_txtdata(txt[1])

ft = open(unfresh_triple, 'w', encoding='utf-8')
fe = open(unfresh_entity, 'w', encoding='utf-8')
# dir1_file = [i.split('.')[0] for i in os.listdir(dir1)]
dir2_file = [i.split('.')[0] for i in os.listdir(dir2)]
for file in os.listdir(dir1):
    if file.endswith('.sh'):
        continue
    start = time.time()
    print('查找%s' % file)
    f1 = open(os.path.join(dir1, file), encoding='utf-8')  # 整个文件都没了
    if file.split('.')[0] not in dir2_file:
        print('%s整个文件都被修改了' % file)
        for line in f1:
            ft.write(line.strip() + '\n')
            fe.write(line.strip().split(' ')[0] + '\n')
        continue
    f1_count = len(open(os.path.join(dir1, file), encoding='utf-8').readlines())
    print(os.path.join(dir1, file), '长度：', f1_count)
    f2 = open(os.path.join(dir2, file), encoding='utf-8')
    f2_count = len(open(os.path.join(dir2, file), encoding='utf-8').readlines())
    print(os.path.join(dir2, file), '长度：', f2_count)
    if f1_count == f2_count:
        # 我赌你的枪里没子弹！
        continue
    batch1, batch2 = [], []
    f1_num, f2_num = 0, 0
    mark1, mark2 = False, False
    while True:
        for line in f1:
            batch1.append(line)
            if len(batch1) == batch_size:
                break
        for line in f2:
            batch2.append(line)
            if len(batch2) == batch_size:
                break
        f1_num += len(batch1)
        f2_num += len(batch2)
        print(
            'file:%s, %d/%d, %d/%d, spend time: %.4f' % (file, f1_num, f1_count, f2_num, f2_count, time.time() - start))
        # 遍历到末尾了
        if len(batch1) < batch_size:
            mark1 = True
        if len(batch2) < batch_size:
            mark2 = True
        batch1 = [i for i in batch1 if i not in batch2]
        ft.write('\n'.join([line.strip() for line in list(batch1)]) + '\n')
        fe.write('\n'.join([line.strip().split(' ')[0] for line in list(batch1)]) + '\n')
        batch1, batch2 = [], []
        # 如果遍历到末尾了，则退出循环
        if mark1 or mark2:
            break
    # 如果第一个文档还有内容
    if not mark1:
        batch1 = f1.read()
        ft.write(batch1)
        entities = set([line.strip().split(' ')[0] for line in batch1])
        fe.write('\n'.join(list(entities)) + '\n')
