import pickle
import json

from sklearn.model_selection import train_test_split

sanwen = './data/sanwen.pkl'
yunwen = {
    'shi': './data/shi.pkl',
    'ci': './data/ci.pkl',
    'shijing': './data/shijing.pkl',
    'chuci': './data/chuci.pkl',
    'yuanqu': './data/yuanqu.pkl'
}


def load_bi_data(ratio):
    all_sw_x = []
    all_sw_y = []

    with open(sanwen, 'rb') as inpt:
        sw_data = pickle.load(inpt)
        print('total sanwen:', len(sw_data))
        for i in sw_data:
            all_sw_x.append(i['context'])
            all_sw_y.append('sanwen')

    X_train, X_test, y_train, y_test = train_test_split(all_sw_x, all_sw_y, test_size=0.1, random_state=42)


    for div in yunwen.keys():
        yw_x = []
        yw_y = []
        with open(yunwen[div], 'rb') as inpt:
            yw_data = pickle.load(inpt)
            print('total ' + div + ':' + len(yw_data))
            for i in yw_data:
                yw_x.append(i['context'])
                yw_y.append('yunwen')

        xt, xte, yt, yte = train_test_split(yw_x, yw_y, test_size=0.1, random_state=42)
        X_train += xt
        X_test += xte
        y_train += yt
        y_test += yte

    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=ratio, random_state=42)
    return X_train, X_test, y_train, y_test


def load_cls_data(ratio):
    all_sw_x = []
    all_sw_y = []

    with open(sanwen, 'rb') as inpt:
        sw_data = pickle.load(inpt)
        for i in sw_data:
            all_sw_x.append(i['context'])
            all_sw_y.append('sanwen')

    X_train, X_test, y_train, y_test = train_test_split(all_sw_x, all_sw_y, test_size=0.1, random_state=42)

    for div in yunwen.keys():
        yw_x = []
        yw_y = []
        with open(yunwen[div], 'rb') as inpt:
            yw_data = pickle.load(inpt)
            for i in yw_data:
                yw_x.append(i['context'])
                yw_y.append(i['div'])

        xt, xte, yt, yte = train_test_split(yw_x, yw_y, test_size=0.1, random_state=42)
        if div != 'shijing' and div != 'chuci':
            xt, _, yt, _ = train_test_split(xt, yt, test_size=ratio, random_state=42)

        X_train += xt
        X_test += xte
        y_train += yt
        y_test += yte

    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=ratio, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    with open(sanwen, 'rb') as inpt:
        sw_data = pickle.load(inpt)
        print('total sanwen:', len(sw_data))

    for div in yunwen.keys():
        with open(yunwen[div], 'rb') as inpt:
            yw_data = pickle.load(inpt)
            print('total', div, ':', len(yw_data))
