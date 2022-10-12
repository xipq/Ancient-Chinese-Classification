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


def process_sw(dir_pool, sav_dir):
    t = []
    for corpus in dir_pool:
        # list, json, pickle, hdf5 all ok
        with open('C:/Users/X/Desktop/CLS/data/sanwen/pool/' + corpus, 'r', encoding='utf-8') as inp:
            inpt = json.load(inp)
            for instance in inpt['类']:
                for txt in instance["book"]["tb_bookviews"]["bookviews"]:
                    text = txt["content"]["tb_bookview"]["cont"]
                    if text == None:
                        continue

                    for lines in text.split('。'):
                        raw = [i.strip('\n').strip('\r').strip('。').strip('<|p|>|/|\u3000|&|q|u|o|t|s|r|n|g|，|（|）|“|”|{|}|：|。| ？| ！|、|(|)|\n') for
                         i in lines]
                        raw = list(filter(None, raw))
                        if len(raw) <= 7:
                            continue

                        tmp = {}
                        tmp['class'] = 'sanwen'
                        tmp['context'] = raw
                        t.append(tmp)

    with open(sav_dir, 'wb') as outp:
        pickle.dump(t, outp)


if __name__ == '__main__':
    sw = os.listdir('C:/Users/X/Desktop/CLS/data/sanwen/pool/')
    sw_sav = './data/yunwen/sanwen.pkl'
    process_sw(sw, sw_sav)