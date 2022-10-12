# naive configs

import argparse

parser = argparse.ArgumentParser(description='model args')

parser.add_argument('--model', type=str, default='BERT', help='Type of pretrained model')
parser.add_argument('--lr', type=int, default=1e-5, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='num epochs')
parser.add_argument('--mode', type=str, default='cls', help='cls modes')

args = parser.parse_args()

# Macros
SEQUENCE_LENGTH = 60
EPOCHS = 30
EARL_STOPPING_PATIENCE = 10
REDUCE_RL_PATIENCE = 5
BATCH_SIZE = 64

# Dirs
bert_path = './pretrained/bert'
ernie_path = './pretrained/ernie'
roberta_path = './pretrained/roberta'
elec_path = './pretrained/electra'
log_dir = './log'
tf_log_dir = './tflog'