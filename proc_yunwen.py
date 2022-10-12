# Demo of Chinese Paragraph Classification

import json
import os

import numpy
import h5py
import pickle
import pandas as pd

'''
sanwen class: [{'class': 'sanwen', 'context':['我', '是', ...]}, ... ]
yunwen class: [{'class':'yunwen', 'div': 'shi', 'context': ['_', ...]}, ...]
'''


def process_shijing(dir_pool, sav_dir):
    for corpus in dir_pool:
        # list, json, pickle, hdf5 all ok
        t = []
        with open(corpus, 'r', encoding='utf-8') as inp:
            inpt = json.load(inp)
            for instance in inpt:
                for txt in instance["content"]:
                    tmp = {}
                    tmp['div'] = 'shijing'
                    tmp['class'] = 'yunwen'
                    tmp['context'] = [i.strip('\n').strip('\r').strip('。').strip('，|（|）|“|”|{|}|：|。| ？| ！|、|(|)|\n') for
                                      i in txt]
                    tmp['context'] = list(filter(None, tmp['context']))

                    if len(tmp['context']) <= 3:
                        continue

                    t.append(tmp)

        with open(sav_dir, 'wb') as outp:
            pickle.dump(t, outp)


def process_chuci(dir_pool, sav_dir):
    for corpus in dir_pool:
        # list, json, pickle, hdf5 all ok
        t = []
        with open(corpus, 'r', encoding='utf-8') as inp:
            inpt = json.load(inp)
            for instance in inpt:
                for txt in instance["content"]:
                    tmp = {}
                    tmp['div'] = 'chuci'
                    tmp['class'] = 'yunwen'
                    tmp['context'] = [i.strip('\n').strip('\r').strip('。').strip('，|（|）|“|”|{|}|：|。| ？| ！|、|(|)|\n') for
                                      i in txt]
                    tmp['context'] = list(filter(None, tmp['context']))

                    if len(tmp['context']) <= 3:
                        continue

                    t.append(tmp)

        with open(sav_dir, 'wb') as outp:
            pickle.dump(t, outp)


def process_yq(dir_pool, sav_dir):
    for corpus in dir_pool:
        # list, json, pickle, hdf5 all ok
        t = []
        with open(corpus, 'r', encoding='utf-8') as inp:
            inpt = json.load(inp)
            for instance in inpt:
                for txt in instance["paragraphs"]:
                    tmp = {}
                    tmp['div'] = 'yuanqu'
                    tmp['class'] = 'yunwen'
                    tmp['context'] = [i.strip('\n').strip('\r').strip('。').strip('，|（|）|“|”|{|}|：|。| ？| ！|、|(|)|\n') for
                                      i in txt]
                    tmp['context'] = list(filter(None, tmp['context']))

                    if len(tmp['context']) <= 7:
                        continue

                    t.append(tmp)

        with open(sav_dir, 'wb') as outp:
            pickle.dump(t, outp)


def process_shi(dir_pool, sav_dir):
    t = []
    for corpus in dir_pool:
        # list, json, pickle, hdf5 all ok
        with open('C:/Users/X/Desktop/CLS/data/yunwen/tangshi/all/' + corpus, 'r', encoding='utf-8') as inp:
            inpt = json.load(inp)
            for instance in inpt:
                for txt in instance["paragraphs"]:
                    tmp = {}
                    tmp['div'] = 'shi'
                    tmp['class'] = 'yunwen'
                    tmp['context'] = [i.strip('\n').strip('\r').strip('。').strip('，|（|）|“|”|{|}|：|。| ？| ！|、|(|)|\n') for
                                      i in txt]
                    tmp['context'] = list(filter(None, tmp['context']))

                    if len(tmp['context']) <= 7:
                        continue

                    t.append(tmp)

    with open(sav_dir, 'wb') as outp:
        pickle.dump(t, outp)


def process_ci(dir_pool, sav_dir):
    t = []
    for corpus in dir_pool:
        # list, json, pickle, hdf5 all ok
        with open('C:/Users/X/Desktop/CLS/data/yunwen/songci/' + corpus, 'r', encoding='utf-8') as inp:
            inpt = json.load(inp)
            for instance in inpt:
                for txt in instance["paragraphs"]:
                    tmp = {}
                    tmp['div'] = 'ci'
                    tmp['class'] = 'yunwen'
                    tmp['context'] = [i.strip('\n').strip('\r').strip('。').strip('，|（|）|“|”|{|}|：|。| ？| ！|、|(|)|\n') for
                                      i in txt]
                    tmp['context'] = list(filter(None, tmp['context']))

                    if len(tmp['context']) <= 7:
                        continue

                    t.append(tmp)

    print(t)
    with open(sav_dir, 'wb') as outp:
        pickle.dump(t, outp)


if __name__ == '__main__':
    sj = ['./data/yunwen/shijing/shijing.json']
    sj_sav = './data/yunwen/shijing.pkl'
    process_shijing(sj, sj_sav)

    cc = ['./data/yunwen/chuci.json']
    cc_sav = './data/yunwen/chuci.pkl'
    process_chuci(cc, cc_sav)

    yq = ['./data/yunwen/yuanqu.json']
    yq_sav = './data/yunwen/yuanqu.pkl'
    process_yq(yq, yq_sav)

    cs = os.listdir('C:/Users/X/Desktop/CLS/data/yunwen/tangshi/all/')
    cs_sav = './data/yunwen/shi.pkl'
    process_shi(cs, cs_sav)

    ci = os.listdir('C:/Users/X/Desktop/CLS/data/yunwen/songci/')
    c_sav = './data/yunwen/ci.pkl'
    process_ci(ci, c_sav)