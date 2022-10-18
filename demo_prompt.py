# Demo of Chinese Paragraph Classification with Prompting

import os
import json
import tensorflow as tf
from tensorflow import keras

import numpy
import kashgari
import pandas as pd

from config import *
from plot_res import *
from dataloader import load_prompted_data_1, load_prompted_data_2, load_prompted_data_3, load_prompted_data_4
from bypass import Prompt_1, Prompt_2, Prompt_3, Prompt_4
from kashgari.callbacks import EvalCallBack
from kashgari.embeddings import BertEmbedding, TransformerEmbedding

# Manual Seed
tf.random.set_seed(42)

# Setup macros
SEQUENCE_LENGTH = 60
EPOCHS = 15
EARL_STOPPING_PATIENCE = 20
REDUCE_RL_PATIENCE = 5
BATCH_SIZE = 64

# ernie as embedding
ernie = TransformerEmbedding(vocab_path=os.path.join(ernie_path, 'vocab.txt'),
                             config_path=os.path.join(ernie_path, 'bert_config.json'),
                             checkpoint_path=os.path.join(ernie_path, 'bert_model.ckpt'),
                             model_type='bert')

embeddings = [
    ('ERNIE', ernie),
]

model_classes = [
    ('一则[MASK]如下', Prompt_1),
    ('下文是一段[MASK]', Prompt_2),
    ('[MASK]', Prompt_3),
    ('有[MASK]云', Prompt_4),
]

# train and eval
for embed_name, embed in embeddings:
    for model_name, MOEDL_CLASS in model_classes:
        run_name = f"{embed_name}_{model_name}"

        if os.path.exists(log_dir):
            logs = json.load(open(log_dir, 'r'))
        else:
            logs = {}
        if logs:
            display.clear_output(wait=True)
            show_plot(logs)

        if embed_name in logs and model_name in logs[embed_name]:
            print(f"Skip {run_name}, already finished")
            continue
        print('=' * 50)
        print(f"\nStart {run_name}")
        print('=' * 50)
        model = MOEDL_CLASS(embed, sequence_length=SEQUENCE_LENGTH)

        early_stop = keras.callbacks.EarlyStopping(patience=EARL_STOPPING_PATIENCE)
        reduse_lr_callback = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                               patience=REDUCE_RL_PATIENCE)

        # Load promped data
        if model_name == '一则[MASK]如下':
            train_x, valid_x, train_y, valid_y = load_prompted_data_1()
        elif model_name == '下文是一段[MASK]':
            train_x, valid_x, train_y, valid_y = load_prompted_data_2()
        elif model_name == '[MASK]':
            train_x, valid_x, train_y, valid_y = load_prompted_data_3()
        elif model_name == '有[MASK]云':
            train_x, valid_x, train_y, valid_y = load_prompted_data_4()

        eval_callback = EvalCallBack(kash_model=model,
                                     x_data=valid_x,
                                     y_data=valid_y,
                                     truncating=True,
                                     step=1)

        tf_board = keras.callbacks.TensorBoard(
            log_dir=os.path.join(tf_log_dir, run_name),
            update_freq=1000
        )

        callbacks = [early_stop, reduse_lr_callback, eval_callback, tf_board]

        model.fit(train_x,
                  train_y,
                  valid_x,
                  valid_y,
                  callbacks=callbacks,
                  epochs=EPOCHS)

        if embed_name not in logs:
            logs[embed_name] = {}

        logs[embed_name][model_name] = eval_callback.logs

        with open(log_dir, 'w') as f:
            f.write(json.dumps(logs, indent=2))
        del model

# Visualize Results
with open(log_dir, 'r') as f:
    full_data = json.loads(f.read())

show_plot(full_data)

# Statictics
pd_data = []
for embed, models in full_data.items():
    for model_name, epochs in models.items():
        max_score = 0
        max_epoch = 0
        for epoch_index, epoch_data in enumerate(epochs):
            if epoch_data['f1-score'] > max_score:
                max_score = epoch_data['f1-score']
                max_epoch = epoch_index
        pd_data.append({
            'Pretrained': embed,
            'CLS Head': model_name,
            'Best F-1': round(max_score * 100, 2),
            'Best Epoch': max_epoch
        })
df = pd.DataFrame(pd_data)
print(df)
