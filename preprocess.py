'''
@author: Zhouhong Gu
@data: 2021/10/20
'''

import os
import bz2
import tqdm


def get_data(zip_path, txt_path, parse_path, decompressing, parsing):
    if decompressing:
        if not os.path.exists(txt_path):
            os.mkdir(txt_path)
        for (dirpath, dirnames, files) in os.walk(zip_path):
            for filename in tqdm.tqdm(files):
                try:
                    filepath = os.path.join(dirpath, filename)
                    newfilepath = os.path.join(txt_path, '.'.join(filename.split('.')[:-1]))
                    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
                        for data in iter(lambda: file.read(100 * 1024), b''):
                            new_file.write(data)
                except:
                    pass
    if parsing:
        fw = open(parse_path, 'w', encoding='utf-8')
        for file in tqdm.tqdm(os.listdir(txt_path)):
            f = open(os.path.join(txt_path, file), encoding='utf-8')
            for line in f:
                try:
                    s, p, o, = line.strip().split(' ')[:3]
                    fw.write('\t'.join([s, p, o]) + '\n')
                except:
                    continue


if __name__ == '__main__':
    decompressing = True
    parsing = False
    six = False
    nine = True
    live = False

    if six:
        print('0601')
        zip_path = 'dataset/dbpedia_0601'
        txt_path = 'dataset/dbpedia_0601_txt'
        parse_path = './dataset/dbpedia_0601.raw.txt'
        get_data(zip_path, txt_path, parse_path, decompressing, parsing)

    if nine:
        print('0901')
        zip_path = 'dataset/dbpedia_0901'
        txt_path = 'dataset/dbpedia_0901_txt'
        parse_path = './dbpedia_0901.raw.txt'
        get_data(zip_path, txt_path, parse_path, decompressing, parsing)

    if live:
        print('live')
        zip_path = 'dataset/dbpedia_live'
        txt_path = 'dataset/dbpedia_live_txt'
        parse_path = './dataset/dbpedia_live.raw.txt'
        get_data(zip_path, txt_path, parse_path, decompressing, parsing)
