'''
@author: Unknown
@update: Zhouhong Gu
@data: 2021/10/13
'''
import random
from pymongo import MongoClient
from datetime import datetime
import os
from bs4 import BeautifulSoup
import requests
import re
import tqdm
import numpy as np
import time
import collections

"""prepare data base"""

# 连接数据库
mongo_cndbpedia = {
    'ip': '47.100.22.113',
    'port': 32017,
    'db': 'cndbpedia',
    'user': 'datareader',
    'password': 'gdmlab@dataer',
    'mechanism': 'SCRAM-SHA-1'
}
mongo_dbpedia = {
    'ip': 'localhost',
    'port': 27017,
    'db': 'dbpedia',
    'user': None,
    'password': None,
    'mechanism': None
}

"""
# 统计平均三元组个数
entities = list(cndbpedia.features.aggregate([{'$sample': {'size': 5000}}]))
l = []
k = 0
for eee in entities:
    eeee = eee['_id']
    ppp = list(cndbpedia.triples.find({'s': eeee}))
    l.append(len(ppp))
    k += 1
    print(k)
print(np.mean(l))
print(np.median(l))
# 25%分位数
print(np.percentile(l, 25))
# 75%分位数
print(np.percentile(l, 75))
"""

# 好像是搞什么时间
# if connectDB:
# counttt = cndbpedia.triples.find().count()
T0 = datetime.strptime('2015-07-24', '%Y-%m-%d')
T01 = datetime.strptime('2016-07-24', '%Y-%m-%d')
T1 = datetime.strptime('2017-07-24', '%Y-%m-%d')
T2 = datetime.strptime('2019-07-24', '%Y-%m-%d')
T3 = datetime.strptime('2020-01-24', '%Y-%m-%d')
T4 = datetime.strptime('2020-06-24', '%Y-%m-%d')
T_list = [T0, T01, T1, T2, T3, T4]
T_now = datetime.strptime('2020-07-24', '%Y-%m-%d')

datapath = './dataset/cndbpedia/result'
# datapath = './dataset/dbpedia'


# 计算两个时间戳之间的时间差
def time_span(strtime1, strtime2):
    '''
    时间差
    '''
    return (datetime.strptime(strtime1, '%Y-%m-%d') - datetime.strptime(strtime2, '%Y-%m-%d')).total_seconds() // 86400


# 文件末尾写一行
def WriteLine(file, lst):
    file.write('\t'.join([str(x) for x in lst]) + '\n')


# 字符串匹配
def Match_patt(patt, sr):
    match = re.search(patt, sr, re.DOTALL | re.MULTILINE)
    return match.group(1) if match else ''


# 获得url网页
def GetPage(url, cookie='', proxy=''):
    try:
        """
		ua = UserAgent()
		ua.update()
		"""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/70.0.3538.102 Safari/537.36',
                   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                   'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
                   'Connection': 'keep-alive'}
        if cookie != '':
            headers['cookie'] = cookie
        if proxy != '':
            proxies = {'http': proxy, 'https': proxy}
            res = requests.get(url, headers=headers, proxies=proxies, timeout=5.0)
        else:
            res = requests.get(url, headers=headers, timeout=5.0)
        content = res.content
        head = content[:min([3000, len(content)])].decode(errors='ignore')
        charset = Match_patt('charset="?([-a-zA-Z0-9]+)', head)
        if charset == '':
            charset = 'utf-8'
        content = content.decode(charset, errors='replace')
    except Exception as e:
        print(e)
        content = ''
    return content


# 将cndbpedia库中的内容更新到random_entities文件中

class KGFreshnessEvaluation:
    def __init__(self, mongo_detail=None):
        if mongo_detail:
            self.connect_mongodb(mongo_detail)

    def connect_mongodb(self, mongo_detail: dict):
        mongo = mongo_detail

        client = MongoClient(mongo['ip'], mongo['port'])
        if mongo['user']:
            client.admin.authenticate(mongo['user'], mongo['password'], mechanism=mongo['mechanism'])
        self.mongo = client[mongo_detail['db']]
        # names = cndbpedia.list_collection_names()

    # 在这一步会生成 'random_entities.txt'
    def data_process(self, random_entities):
        '''
        增量式更新
        将cndbpedia库中的内容更新到random_entities文件中
        '''
        random.seed(321456)
        entities = list(self.mongo.features.aggregate([{'$sample': {'size': 100000}}]))
        print(entities[:10])
        # 实体 + 时间
        already = []
        try:
            with open(random_entities, 'r', encoding='utf-8') as file1:
                for line in file1:
                    lln = line.rstrip('\r\n').split('\t')
                    already.append(lln)
                # 单纯的实体
                processed_ent = set([kkk[0] for kkk in already])
        except:
            print('未找到文件: %s' % random_entities)
            processed_ent = set()
        # 增量式更新
        s_name = '_id' if '/cndbpedia' in datapath else 's'
        print(s_name)
        s_name = 's'
        with open(random_entities, 'a+', encoding='utf-8') as file2:
            for item in tqdm.tqdm(entities):
                if item[s_name] not in processed_ent:
                    if 'timestamp' in item:
                        WriteLine(file2, [item[s_name], item['timestamp']])
                        file2.flush()
                    # WriteLine(file1, zip(fea[0].keys(), fea[0].values()))

    # 根据百度百科的内容，检查entity的实时性
    # 具体有多么不实时主要修改这里的内容
    def Check_update(self, id, time_info, use_timestamp: bool):
        '''
        根据百度百科的内容，检查entity的实时性
        @id: 实体
        @time_info: 实体的时间信息 <上次更新时间，timestamp>
        @use_timestamp: 某种标记
        return: [0 if 不实时 else 1, 网页更新时间 if 0 else 数据库更新时间]
        '''
        url = 'http://baike.baidu.com/item/%s' % id
        # url = 'https://zh.wikipedia.org/wiki/%s' %id
        page = GetPage(url)
        soup = BeautifulSoup(page, 'html.parser')
        try:
            # if not soup.title.text.endswith('_百度百科'):
            #     return 'nonentity'
            # name = soup.title.text.replace('_百度百科', '').lower()
            page = re.sub('[\r\t\n]', ' ', page)
            page = re.sub('[ ]+', ' ', page)
            last_upd_db, timestamp = time_info
            if not use_timestamp:
                patt = '<span class="j-modified-time" style="display:none">(.+?)</span>'
                last_upd_web = Match_patt(patt, page)[1:11]
                if len(last_upd_web) < 2:
                    return 'nonentity'
                last_upd_web = datetime.strptime(last_upd_web, '%Y-%m-%d')
                if timestamp < last_upd_web:
                    return [0, last_upd_web]
                else:
                    return [1, last_upd_web]
            else:
                patt = '"(/historylist/.+?)"'
                url = 'https://baike.baidu.com' + Match_patt(patt, page)
                # url = 'https://zh.wikipedia.org/wiki/%s' %id
                page_hist = GetPage(url)
                for update in re.findall('<td>([0-9-]+) .+?</td>', page_hist):
                    last_upd_web = datetime.strptime(update, '%Y-%m-%d')
                    if last_upd_web < timestamp:
                        if last_upd_web > last_upd_db:
                            return [0, last_upd_web]
                        else:
                            return [1, last_upd_db]
        except:
            return 'nonentity'

    # 检测每个实体是否实时
    # 输入的是<entity, last-upd-db>，输出的是<entity, label, last-upd-db, last-upd-web>
    def update_test(self, fin, fout):
        read_in = []
        output = []
        processed = []
        try:
            with open(fout, 'r', encoding='utf-8') as f1:
                for line in f1:
                    lln = line.rstrip('\r\n').split('\t')
                    processed.append(lln)
            f1.close()
        except:
            processed = []
        last_one = processed[-1][0] if processed else None
        fresh = [int(kk[1]) for kk in processed] if processed else []
        delay = [(datetime.strptime(processed[i][3], '%Y-%m-%d') -
                  datetime.strptime(processed[i][2], '%Y-%m-%d')).days()
                 if not fresh[i] else 0 for i in range(len(fresh))]
        fresh_list = []
        delay_list = []
        for i in range(len(T_list)):
            tt = T_list[i]
            fresh_list.append(
                [fresh[i] for i in range(len(fresh)) if datetime.strptime(processed[i][3], '%Y-%m-%d') > tt])
            delay_list.append(
                [delay[i] for i in range(len(fresh)) if datetime.strptime(processed[i][3], '%Y-%m-%d') > tt])
        with open(fout, 'a+', encoding='utf-8') as f1:
            with open(fin, encoding='utf-8') as f:
                for line in f:
                    lln = line.rstrip('\r\n').replace('_',' ').split('\t')
                    read_in.append(lln)
            start = ([k[0] for k in read_in]).index(last_one) + 1 if last_one else 0
            for ind in range(start, len(read_in)):
                ii = read_in[ind]
                id = ii[0]
                timestamp = datetime.strptime(ii[1][:10], '%Y-%m-%d')
                check = self.Check_update(id, ['', timestamp], False)
                if not (check == 'nonentity'):
                    print("No.%d out of %d is finished" % (ind + 1, len(read_in)), check)
                    time.sleep(2)
                    # 输出的内容
                    # output.append([id, check[0], datetime.date(timestamp), datetime.date(check[1])])
                    output.append(
                        [id, check[0], datetime.date(timestamp), datetime.date(check[1]),
                         time_span(str(check[1].date()), str(timestamp.date())) if check[0] == 0 else 0])
                    f1.write('\t'.join([str(x) for x in output[-1]]) + '\n')
                    f1.flush()
                    fresh.append(check[0])
                    for i in range(len(T_list)):
                        if check[1] < T_list[i]:
                            break
                        else:
                            fresh_list[i].append(check[0])
                            delay_list[i].append((datetime.date(check[1]) - datetime.date(timestamp)).total_seconds()
                                                 // 86400 / 7 if not check[0] else 0)
                    # print("freshness rate: ", [np.mean(fresh_list[ii]) for ii in range(len(fresh_list))])
                    # print("delay:", [np.mean(delay_list[ii]) for ii in range(len(delay_list))])
                    # print("proportion: ", [len(fresh_list[ii]) / len(fresh) for ii in range(len(fresh_list))])
                    # print(np.mean(delay))
                    # print(np.mean(fresh))
                    """
    				for k in range(100, len(fresh)):
    					if 1.96*np.sqrt(np.mean(fresh[:k])*(1-np.mean(fresh[:k]))/len(fresh[:k]))<0.01:
    						print("sample size", k)
    						break
    				"""
                else:
                    print("%s does not exist" % (ii))
        return np.mean(fresh)


KGFreshnessEvaluation = KGFreshnessEvaluation(mongo_cndbpedia)

random_entities = os.path.join(datapath, 'random_entities.txt')
random_entities_fresh = os.path.join(datapath, 'random_entities_fresh.txt')

# KGFreshnessEvaluation.data_process(random_entities)
KGFreshnessEvaluation.update_test(random_entities, random_entities_fresh)


# 把fin中有但是fout中没有的信息进行补全更新，只补全histfreq这一词条，结果保存至fout
# histfreq是什么意思？
def histfreq_process(fin, fout):
    '''
    fout应该只包含了fin的前n条信息
    1. 首先找到这个n在哪里
    2. 从cndbpedia中获得后面信息的具体内容
    3. 把具体内容更新到fout里面
    '''
    data_in = []
    # 增量式更新fout
    processed = []
    with open(fin, 'r', encoding='utf-8') as f1:
        for line in f1:
            lln = line.rstrip('\r\n').split('\t')
            data_in.append(lln)
    try:
        with open(fout, 'r', encoding='utf-8') as ff:
            for line in ff:
                lln = line.rstrip('\r\n').split('\t')
                processed.append(lln)
    except:
        pass
    last_one = processed[-1][0]
    # fin文件中所有的id
    ids = [item[0] for item in data_in]
    # fin文件中所有的更新时间
    last_update = [item[3] for item in data_in]
    # 这个output好像没啥用？
    output = []
    with open(fout, 'a+', encoding='utf-8') as f2:
        for kk in range(ids.index(last_one) + 1, len(ids)):
            fea = list(cndbpedia.features.find({'_id': ids[kk]}))
            if 'histfreq' in fea[0]:
                timestamps_new = fea[0]['timestamp']
                hist_freq = fea[0]['histfreq']
                fresh_new = 0 if datetime.strptime(last_update[kk], '%Y-%m-%d') > timestamps_new else 1
                output.append([ids[kk], fresh_new, datetime.date(timestamps_new), last_update[kk], hist_freq])
                f2.write('\t'.join([str(x) for x in output[-1]]) + '\n')
                f2.flush()
            else:
                continue
    f2.close()


# 清洗乱七八糟的html标记
def MakeALabel(z):
    '''
    清洗乱七八糟的html标记
    '''
    z = str(z).strip().replace('\xa0', '').replace('&nbsp;', ' ')
    z = re.sub('[\r\n\t]', '', z)
    z = re.sub('<sup>.+?</sup>', '', z)
    z = re.sub('<a .+?>', '*a*', z).replace('</a>', '*/a*')
    z = re.sub('<br/?>', '\n', z)
    z = re.sub('<.+?>', '', z)
    z = z.replace('*a*', '<a>').strip()
    z = z.replace('*/a*', '</a>').strip()
    z = z.replace('<a></a>', '').strip()
    return z


def extract_content(con, fout):
    with open(fout, 'a+', encoding='utf-8') as fff:
        fff.write('\t'.join([str(x) for x in con]) + '\n')
        fff.flush()


# 从百度百科中爬取有关id的内容，这里的id应该是个实体的名字
# 返回：<创建时间, hist_freq, 编辑次数, 浏览次数, inner_links, all_links, 段落数, 内容数, 实体字符长度>
# 这里还返回了另一个文件：'random_entities.txt'，大概内容：<entity, desc, <triple1>, <triple2>, ..., <tripleN>>
def extract_baike(entity):
    '''
    从百科中爬取id有关的内容
    '''
    try:
        url = 'http://baike.baidu.com/item/%s' % entity
        page = GetPage(url)
        page = re.sub('[\r\t\n]', ' ', page)
        page = re.sub('[ ]+', ' ', page)
        soup = BeautifulSoup(page, 'html.parser')
        if not soup.title.text.endswith('_百度百科'):
            return None
        patt = '"(/historylist/.+?)"'
        url2 = 'https://baike.baidu.com' + Match_patt(patt, page)
        page2 = GetPage(url2)

        def find_triples(sssoup):
            infobox = sssoup.find('div', 'basic-info')
            kvs = []
            if infobox is not None:
                ks = infobox.find_all('dt')
                vs = infobox.find_all('dd')
                for k, v in zip(ks, vs):
                    k = k.text.strip().replace('\xa0', '')
                    v = MakeALabel(v)
                    kvs.append((k, v))
            summ = soup.find('div', 'lemma-summary')
            paras = [MakeALabel(x) for x in summ.find_all('div', 'para')]
            paras = [x for x in paras if x != '']
            if len(paras) > 0:
                desc = '\n'.join(paras)
                kvs.append(('DESC', desc))
            return kvs

        total_edit = int(Match_patt('共被编辑([0-9]+)次', page2)) if Match_patt('共被编辑([0-9]+)次', page2) else 0
        lemmapv = Match_patt('newLemmaIdEnc:"(.+?)"', page)
        lemmapg = GetPage('http://baike.baidu.com/api/lemmapv?id=%s' % lemmapv)
        view_times = int(Match_patt('([0-9]+)', lemmapg))
        inner_links = page.count('href="/item')
        all_links = page.count('href="')
        paras = [re.sub('<.+?>', '', MakeALabel(x)) for x in soup.find_all('div', 'para')]
        paras = [x for x in paras if x != '']
        page_len = sum(len(x) for x in paras)
        summ = soup.find('div', 'lemma-summary')
        paras = [re.sub('<.+?>', '', MakeALabel(x)) for x in summ.find_all('div', 'para')]
        paras = [x for x in paras if x != '']
        content_len = sum(len(x) for x in paras)
        tk = Match_patt('tk.+?=.+?"(.+?)";', page2)
        lemmaId = re.search('lemmaId.+?=.+?([0-9]+);', page2).group(1)
        pgnum = (total_edit + 24) // 25
        for zz in re.findall('<td>([0-9-]+) .+?</td>', page2):
            if datetime.strptime(zz, '%Y-%m-%d') > T_now:
                total_edit -= 1
        # print(tk, lemmaId)
        url3 = 'https://baike.baidu.com/api/wikiui/gethistorylist?tk=%s&lemmaId=%s&from=%d&count=1&size=25' % (
            tk, lemmaId, pgnum)
        page3 = GetPage(url3)
        if not re.findall('"createTime":([0-9]+),', page3):
            return None
        extract_content([id, ''.join([str(x) for x in paras]), "#".join([str(x) for x in find_triples(soup)])],
                        'entities_content.txt')
        T_create = re.findall('"createTime":([0-9]+),', page3)[-1]
        T_create = datetime.fromtimestamp(int(T_create))
        hist_freq = total_edit / ((T_now - T_create).total_seconds() // 86400 / 7)
        return [datetime.date(T_create), hist_freq, total_edit, view_times, inner_links, all_links, page_len,
                content_len, len(id)]
    except:
        return None


# 获取一个实体的创建时间与所有修改时间
def history_list(entity):
    try:
        url = 'http://baike.baidu.com/item/%s' % entity
        page = GetPage(url)
        page = re.sub('[\r\t\n]', ' ', page)
        page = re.sub('[ ]+', ' ', page)
        patt = '"(/historylist/.+?)"'
        url2 = 'https://baike.baidu.com' + Match_patt(patt, page)
        page2 = GetPage(url2)
        total_edit = int(Match_patt('共被编辑([0-9]+)次', page2)) if Match_patt('共被编辑([0-9]+)次', page2) else 0
        tk = Match_patt('tk.+?=.+?"(.+?)";', page2)
        lemmaId = re.search('lemmaId.+?=.+?([0-9]+);', page2).group(1)
        pgnum = (total_edit + 24) // 25
        for zz in re.findall('<td>([0-9-]+) .+?</td>', page2):
            if datetime.strptime(zz, '%Y-%m-%d') > T_now:
                total_edit -= 1
        # print(tk, lemmaId)
        url3 = 'https://baike.baidu.com/api/wikiui/gethistorylist?tk=%s&lemmaId=%s&from=%d&count=1&size=25' % (
            tk, lemmaId, pgnum)
        page3 = GetPage(url3)
        Time_change = re.findall('"createTime":([0-9]+),', page3)
        L = []
        for i in Time_change:
            st = datetime.fromtimestamp(int(i))
            if st < T_now:
                L.append(datetime.date(st))
        return L
    except:
        return None


# 把fin中有但是fout中没有的信息进行补全更新，只补全所有的历史修改时间与创建时间，结果保存至fout
def get_history_list(fin, fout):
    ene = []
    processed = []
    with open(fin, 'r', encoding='utf-8') as f1:
        for line in f1:
            lln = line.rstrip('\r\n').split('\t')
            ene.append(lln[0])
    try:
        with open(fout, 'r', encoding='utf-8') as ff:
            for line in ff:
                lln = line.rstrip('\r\n').split('\t')
                processed.append(lln[0])
    except:
        print('未找到文件：%s' % fout)
    last_one = processed[-1] if processed else None
    start = ene.index(last_one) + 1 if last_one else 0
    print(start)
    with open(fout, 'a+', encoding='utf-8') as f2:
        for kk in range(start, len(ene)):
            kkk = ene[kk]
            hlist = history_list(kkk)
            if not hlist:
                continue
            output = []
            output.append(kkk)
            print(kk, "out of", len(ene), "is finished")
            output.extend(hlist)
            f2.write('\t'.join([str(x) for x in output]) + '\n')
            f2.flush()
            time.sleep(1.5)


# get_history_list(os.path.join(datapath, 'main_ent_info.txt'), os.path.join(datapath, 'ent_hist.txt'))


# 把fin中有但是fout中没有的信息进行补全更新，要补全的内容写在函数里面的注释了，结果保存至fout
def info_extract_web(fin, fout):
    '''
    fout输出内容：<entity, 是否实时, timestamp, 最后更新时间, 创建时间, hist_freq, 编辑次数, 浏览次数, inner_links, all_links, 段落数, 内容数, 实体字符长度>
    '''
    data_in = []
    processed = []
    with open(fin, 'r', encoding='utf-8') as f1:
        for line in f1:
            lln = line.rstrip('\r\n').split('\t')
            data_in.append(lln)
    with open(fout, 'r', encoding='utf-8') as ff:
        for line in ff:
            lln = line.rstrip('\r\n').split('\t')
            processed.append(lln)
    last_one = processed[-1][0] if processed else None
    ids = [item[0] for item in data_in]
    start = ids.index(last_one) + 1 if last_one else 0
    last_update = [item[3] for item in data_in]
    with open(fout, 'a+', encoding='utf-8') as f2:
        for kk in range(start, len(ids)):
            fea = list(cndbpedia.features.find({'_id': ids[kk]}))
            info = extract_baike(ids[kk])
            if not info:
                continue
            timestamps_new = fea[0]['timestamp'] if 'timestamp' in fea[0] else datetime.strptime(data_in[kk][2],
                                                                                                 '%Y-%m-%d')
            fresh_new = 0 if datetime.strptime(last_update[kk], '%Y-%m-%d') > timestamps_new else 1
            output = [ids[kk], fresh_new, datetime.date(timestamps_new), last_update[kk]] + info
            f2.write('\t'.join([str(x) for x in output]) + '\n')
            f2.flush()
            time.sleep(2)
    f2.close()


# # extract_baike("成龙")
# info_extract_web(os.path.join(datapath, 'random_entities_fresh1.txt'),
#                  os.path.join(datapath, 'random_entities_info.txt'))


# 检验：是否历史更新次数越多的实体，越不实时
def precision_recall(fin):
    '''
    'random_entities_freq1.txt'应该是个什么模型的预测结果吧？
    但这个文件里的信息应该不重要
    这个函数是个检测函数
    检测目标：
        是不是历史更新时间越多的实体，可能越不实时？
    '''
    fresh_list = []
    fresh = []
    lastupdate = []
    hist_freq = []
    with open(fin, 'r', encoding='utf-8') as f1:
        for line in f1:
            lln = line.rstrip('\r\n').split('\t')
            # 是否实时
            fresh.append(int(lln[1]))
            # 最后更新时间
            lastupdate.append(datetime.strptime(lln[3], '%Y-%m-%d'))
            # 所有历史更新时间
            hist_freq.append(eval(lln[5]))
    f1.close()
    for i in range(len(T_list)):
        tt = T_list[i]
        # 最后更新时间大于某个时间戳的话就放入fresh_list中
        fresh_list.append([i for i in range(len(fresh)) if lastupdate[i] > tt])
    # 这里大概的意思应该是——历史更新时间越多的，就越不实时
    # 没次取最频繁更新的20%的数据，假定这些数据是不实时的，然后比对一下真实结果
    length = [round(0.2 * ii * len(fresh)) for ii in range(1, 6)]
    sort_freq = sorted(range(len(hist_freq)), reverse=True, key=lambda k: hist_freq[k])
    predict_list = []
    # 这里好像得到了什么预测的结果
    predict_list.append(sort_freq[:length[0]])
    for ll in range(1, len(length)):
        predict_list.append(sort_freq[length[ll - 1]:length[ll]])
    # TP = [len([ind for ind in predict_list[k] if ind in fresh_list[k]]) for k in range(len(fresh_list))]
    # FN = [len([ind for ind in fresh_list[k] if ind not in predict_list[k]]) for k in range(len(fresh_list))]
    # precision = [TP[k] / len(predict_list[k]) for k in range(len(fresh_list))]
    # recall = [TP[k] / (TP[k] + FN[k]) for k in range(len(fresh_list))]
    f_list = [np.mean([fresh[iii] for iii in predict]) for predict in predict_list]
    # print(precision, recall, f_list)
    print(np.mean(fresh))
    # return precision, recall


# precision_recall(os.path.join(datapath,'random_entities_freq1.txt'))

# 针对单个实体的所有三元组进行实时性评估
def triple_test(s, tri):
    '''
    对tri中的所有三元组进行评估，是否实时
    '''
    # 获得所有三元组
    triples = tri.split("#")
    content = [str(MakeALabel(item).replace('''\\n''', "")) for item in triples]
    triples_out = []
    try:
        for kkk in content:
            (p, o) = eval(kkk)
            o_db = list(cndbpedia.triples.find({'s': s, "p": p}))
            # 为什么这里没找到o_db就直接跳过了？
            if not o_db:
                continue
            # 为什么这里要取o_db[0]？
            ddd = re.sub('''<.?a.?>''', '', MakeALabel(o_db[0]['o']))
            eee = re.sub('''<.?a.?>''', '', MakeALabel(o))
            if ddd == eee:
                triples_out.append([(s, p, eee), 1])
            else:
                triples_out.append([(s, p, eee), 0])
        return triples_out
    except:
        return None


# 三元组级别的实时性评估
def triple_level(fin, fout):
    '''
    fout: <三元组, 是否实时>
    '''
    data_in = []
    processed = []
    with open(fin, 'r', encoding='utf-8') as f1:
        for line in f1:
            lln = line.rstrip('\r\n').split('\t')
            data_in.append(lln)
    try:
        with open(fout, 'r', encoding='utf-8') as ff:
            for line in ff:
                lln = line.rstrip('\r\n').split('\t')
                processed.append(lln)
    except:
        print('未找到文件：%s' % fout)
    (last_one, p, o) = eval(processed[-1][0]) if processed else None
    ids = [item[0] for item in data_in]
    start = ids.index(last_one) + 1 if last_one else 0
    print(start)
    with open(fout, 'a+', encoding='utf-8') as f2:
        for indd in range(start, len(data_in)):
            item = data_in[indd]
            if len(item[2]):
                tri_status = triple_test(item[0], item[2])
                if tri_status:
                    for ele in tri_status:
                        f2.write('\t'.join([str(x) for x in ele]) + '\n')
                        f2.flush()


# triple_level(os.path.join(datapath,'entities_content.txt'), os.path.join(datapath,'main_triple_info.txt'))

# 针对单个属性的所有三元组进行实时性评估
def property_test(s, tri):
    # triples = tri.split("#")
    content = [str(MakeALabel(item).replace('''\\n''', "")) for item in tri]
    output = dict()
    try:
        for kkk in content:
            (p, o) = eval(kkk)
            if p == 'BaiduCARD':
                p = 'DESC'
            if p == 'BaiduTAG':
                continue
            existt = list(cndbpedia.triples.find({'s': s}))
            if not existt:
                return None
            o_db = list(cndbpedia.triples.find({'s': s, "p": p}))
            if not o_db:
                output[p] = 0
                continue
            ddd = [re.sub('''<.?a.?>''', '', MakeALabel(o_db[k]['o'])) for k in range(len(o_db))]
            eee = re.sub('''<.?a.?>''', '', MakeALabel(o))
            if eee in ddd:
                output[p] = 1
            else:
                output[p] = 0
        return output
    except:
        return None


#
def find_property(fin, fout):
    data_in = []
    processed = []
    with open(fin, 'r', encoding='utf-8') as f1:
        pas = 0
        ttt = 0
        for line in f1:
            ttt += 1
            if ttt < 5000:
                continue
            if len(data_in) + 1 % 1000 == 0:
                if pas < 9000:
                    pas += 1
                    continue
                else:
                    pas = 0
            lln = line.rstrip('\r\n').split('\t')
            data_in.append(lln)
            if len(data_in) == 10000000:
                break
    # """
    with open(fout, 'r', encoding='utf-8') as ff:
        for line in ff:
            lln = line.rstrip('\r\n').split('\t')
            processed.append(lln)
    last_one = processed[-1][0] if processed else None
    ids = [item[0] for item in data_in]
    # """
    start = 0
    with open(fout, 'a+', encoding='utf-8') as f2:
        for indd in range(start, len(data_in)):
            item = data_in[indd]
            ooo = property_test(item[0], [(item[1], item[2])])
            if ooo:
                output = []
                output.append(item[0])
                output.extend([(kk, ooo[kk]) for kk in ooo.keys()])
                f2.write('\t'.join([str(x) for x in output]) + '\n')
                f2.flush()


# find_property(os.path.join(datapath,'baike_triples.txt'), os.path.join(datapath,'property.txt'))

def property_statis(fin, fout):
    data_in = []
    with open(fin, 'r', encoding='utf-8') as f1:
        for line in f1:
            lln = line.rstrip('\r\n').split('\t')
            data_in.append(lln)
    statis = collections.defaultdict(lambda: [0, 0])
    stat = collections.defaultdict(float)
    with open(fout, 'w', encoding='utf-8') as f2:
        for indd in range(0, len(data_in)):
            item = data_in[indd][1:]
            for i in range(len(item)):
                (p, value) = eval(item[i])
                statis[p][0] += value
                statis[p][1] += 1
        qq1 = 0
        qq2 = 0
        for kkk in statis:
            stat[kkk] = statis[kkk][0] / statis[kkk][1]
            qq1 += statis[kkk][0]
            qq2 += statis[kkk][1]
        print(qq1 / qq2)

        lll = sorted(stat.items(), key=lambda kv: [kv[1], kv[0]])
        for output in lll:
            f2.write('\t'.join([str(x) for x in output]) + '\n')
            f2.flush()

# property_statis(os.path.join(datapath,'property.txt'), os.path.join(datapath,'prop_sort.txt'))
