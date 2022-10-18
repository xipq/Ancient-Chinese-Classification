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
            print('total ', div, ':' , len(yw_data))
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


def trans(cls):
    if cls == 'sanwen':
        return '文'
    if cls == 'shi':
        return '诗'
    if cls == 'ci':
        return '词'
    if cls == 'shijing':
        return '经'
    if cls == 'chuci':
        return '辞'
    if cls == 'yuanqu':
        return '曲'


def load_prompted_data_1():
    '''
    一则[mask]如下 :
    '''
    train_x, valid_x, train_y, valid_y = load_cls_data(0.95)

    for i in range(0, len(train_x)):
        cls = trans(train_y[i])
        prompt = ['一','则','[MASK]','如','下',':']
        train_x[i] = prompt + train_x[i]
        train_y[i] = cls

    for i in range(0, len(valid_x)):
        cls = trans(valid_y[i])
        prompt = ['一','则','[MASK]','如','下',':']
        valid_x[i] = prompt + valid_x[i]
        valid_y[i] = cls

    return train_x, valid_x, train_y, valid_y


def load_prompted_data_2():
    '''
    下文是一段[mask] :
    '''
    train_x, valid_x, train_y, valid_y = load_cls_data(0.95)

    for i in range(0, len(train_x)):
        cls = trans(train_y[i])
        prompt = ['下', '文', '是', '一', '段', '[MASK]', ':']
        train_x[i] = prompt + train_x[i]
        train_y[i] = cls

    for i in range(0, len(valid_x)):
        cls = trans(valid_y[i])
        prompt = ['下', '文', '是', '一', '段', '[MASK]', ':']
        valid_x[i] = prompt + valid_x[i]
        valid_y[i] = cls

    return train_x, valid_x, train_y, valid_y


def load_prompted_data_3():
    '''
    [mask] :
    '''
    train_x, valid_x, train_y, valid_y = load_cls_data(0.95)

    for i in range(0, len(train_x)):
        cls = trans(train_y[i])
        prompt = ['[MASK]', ':']
        train_x[i] = prompt + train_x[i]
        train_y[i] = cls

    for i in range(0, len(valid_x)):
        cls = trans(valid_y[i])
        prompt = ['[MASK]', ':']
        valid_x[i] = prompt + valid_x[i]
        valid_y[i] = cls

    return train_x, valid_x, train_y, valid_y


def load_prompted_data_4():
    '''
    有 [mask] 云：
    '''
    train_x, valid_x, train_y, valid_y = load_cls_data(0.95)

    for i in range(0, len(train_x)):
        cls = trans(train_y[i])
        prompt = ['有', '[MASK]', '云', ':']
        train_x[i] = prompt + train_x[i]
        train_y[i] = cls

    for i in range(0, len(valid_x)):
        cls = trans(valid_y[i])
        prompt = ['有', '[MASK]', '云', ':']
        valid_x[i] = prompt + valid_x[i]
        valid_y[i] = cls

    return train_x, valid_x, train_y, valid_y



if __name__ == '__main__':

    with open(sanwen, 'rb') as inpt:
        sw_data = pickle.load(inpt)
        print('total sanwen:', len(sw_data))

    for div in yunwen.keys():
        with open(yunwen[div], 'rb') as inpt:
            yw_data = pickle.load(inpt)
            print('total', div, ':', len(yw_data))
