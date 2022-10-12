import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython import display

sns.set_style("whitegrid")


def get_style(name):
    linestyle = None
    if 'CRF' in name:
        linestyle = 'dashed'
    return linestyle


def show_plot(results, target_metric='f1-score'):
    plt.rcParams['figure.dpi'] = 300
    row = len(results) // 2

    plt.figure(figsize=(10, 10))

    index = 1

    for embed_name, model_all in results.items():
        plt.subplot(2, 2, index) 
        index += 1

        for model_name, model_his in model_all.items():
            datas = [i[target_metric] for i in model_his]

            plt.plot(datas, label=model_name, linestyle=get_style(model_name))

        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(target_metric)

        plt.title(f'{embed_name} {target_metric}')
        plt.ylim(None, 1.0)


    plt.savefig('./img_test.svg', dpi=300)
    plt.show()

if __name__ == '__main__':
    with open('./log', 'r') as f:
        full_data = json.loads(f.read())

    show_plot(full_data)