# Demo of Chinese Paragraph Classification

import os
import json
import tensorflow as tf
from tensorflow import keras

import numpy
import kashgari
import pandas as pd

from config import *
from plot_res import *
from dataloader import load_bi_data, load_cls_data
from bypass import FC_Model
from kashgari.tasks.classification import BiGRU_Model, BiLSTM_Model, CNN_Model, CNN_Attention_Model, CNN_GRU_Model, \
    CNN_LSTM_Model
from kashgari.callbacks import EvalCallBack
from kashgari.embeddings import BertEmbedding, TransformerEmbedding

# Manual Seed
tf.random.set_seed(42)

if args.mode == 'binary':
    train_x, valid_x, train_y, valid_y = load_bi_data(0.95)
elif args.mode == 'cls':
    train_x, valid_x, train_y, valid_y = load_cls_data(0.95)

# Setup macros
SEQUENCE_LENGTH = 60
EPOCHS = 8
EARL_STOPPING_PATIENCE = 10
REDUCE_RL_PATIENCE = 5
BATCH_SIZE = 64

# roberta as embedding
roberta = TransformerEmbedding(vocab_path=os.path.join(roberta_path, 'vocab.txt'),
                               config_path=os.path.join(roberta_path, 'bert_config.json'),
                               checkpoint_path=os.path.join(roberta_path, 'bert_model.ckpt'),
                               model_type='bert')

# ernie as embedding
ernie = TransformerEmbedding(vocab_path=os.path.join(ernie_path, 'vocab.txt'),
                             config_path=os.path.join(ernie_path, 'bert_config.json'),
                             checkpoint_path=os.path.join(ernie_path, 'bert_model.ckpt'),
                             model_type='bert')

# electra as embeddingg
electra = TransformerEmbedding(vocab_path=os.path.join(elec_path, 'vocab.txt'),
                             config_path=os.path.join(elec_path, 'base_discriminator_config.json'),
                             checkpoint_path=os.path.join(elec_path, 'electra_180g_base.ckpt'),
                             model_type='electra')

# vanilla bert embedding
bert = BertEmbedding(bert_path)

embeddings = [
    ('BERT-base', bert),
    ('ERNIE', ernie),
    ('RoBERTa-wwm-ext', roberta),
    ('electra', electra),
]

model_classes = [
    ('Direct', FC_Model),
    ('BiLSTM', BiLSTM_Model),
    ('CNN_Attention', CNN_Attention_Model),
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
