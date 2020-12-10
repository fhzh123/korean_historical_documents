# Import Module
import os
import math
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch import optim
from torch.utils.data import DataLoader

# Import Custom Module
from .dataset import CustomDataset, PadCollate
from .model.transformer.Models import PTransformer, Transformer
# from .model.rnn import Encoder, Decoder, Seq2Seq
from .model.transformer.Optim import Ralamb, WarmupLinearSchedule
from .training_module import model_training
from .utils import accuracy

def testing(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    print('Data Load & Setting!')
    if args.dataset == 'total_data':
        with open(os.path.join(args.save_path, 'preprocessed_data.json'), 'r') as f:
            data_ = json.load(f)
            hj_train_indices = data_['train_hanja_indices']
            hj_valid_indices = data_['valid_hanja_indices']
            kr_train_indices = data_['train_korean_indices']
            kr_valid_indices = data_['valid_korean_indices']
            king_train_indices = data_['king_ids_hk_train']
            king_valid_indices = data_['king_ids_hk_valid']
            hj_word2id = data_['hanja_word2id']
            kr_word2id = data_['korean_word2id']
            src_vocab_num = len(hj_word2id.keys())
            trg_vocab_num = len(kr_word2id.keys())

    elif args.dataset == 'normal_data':
        with open(os.path.join(args.save_path, 'nmt_processed.pkl'), 'rb') as f:
            data_ = pickle.load(f)
            hj_test_indices = data_['hj_test_indices']
            kr_test_indices = data_['kr_test_indices']
            king_test_indices = data_['king_test_indices']
            hj_word2id = data_['hj_word2id']
            kr_word2id = data_['kr_word2id']
            src_vocab_num = len(hj_word2id.keys())
            trg_vocab_num = len(kr_word2id.keys())
            del data_