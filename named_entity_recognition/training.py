# Import Module
import os
import math
import time
import pickle
import argparse
import itertools
import numpy as np
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
from .model.transformer import Transformer_model
from .optimizer import Ralamb, WarmupLinearSchedule
from .module import train_model

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    print('Data Load & Setting!')
    with open(os.path.join(args.save_path, 'ner_processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        hj_train_indices = data_['hj_train_indices']
        hj_valid_indices = data_['hj_valid_indices']
        hj_test_indices = data_['hj_test_indices']
        ner_train_indices = data_['ner_train_indices']
        ner_valid_indices = data_['ner_valid_indices']
        ner_test_indices = data_['ner_test_indices']
        king_train_indices = data_['king_train_indices']
        king_valid_indices = data_['king_valid_indices']
        king_test_indices = data_['king_test_indices']
        word2id = data_['hj_word2id']
        id2word = data_['hj_word2id']
        src_vocab_num = len(word2id.keys())
        del data_

    with open(os.path.join(args.save_path, 'hj_emb_mat.pkl'), 'rb') as f:
        emb_mat = pickle.load(f)

    dataset_dict = {
        'train': CustomDataset(hj_train_indices, ner_train_indices, king_train_indices,
                            min_len=args.min_len, max_len=args.max_len),
        'valid': CustomDataset(hj_valid_indices, ner_valid_indices, king_valid_indices,
                            min_len=args.min_len, max_len=args.max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True)
    }
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    print("Instantiating models...")
    model = Transformer_model(src_vocab_num=src_vocab_num, pad_idx=args.pad_idx, bos_idx=args.bos_idx, 
                              eos_idx=args.eos_idx, d_model=args.d_model, d_embedding=args.d_embedding, 
                              n_head=args.n_head, dim_feedforward=args.dim_feedforward, n_layers=args.n_layers, 
                              dropout=args.dropout, baseline=args.baseline, device=device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=len(dataloader_dict['train'])*3, 
                                     t_total=len(dataloader_dict['train'])*args.num_epoch)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    model.to(device)

    #===================================#
    #============DWE setting============#
    #===================================#

    if not args.baseline:
        for i in range(len(emb_mat)):
            model.src_embedding.token_dict[i] = nn.Embedding(src_vocab_num, args.d_embedding, padding_idx=0).to(device)
            for word, id_ in word2id.items():
                try:
                    model.src_embedding.token_dict[i].token.weight.data[id_] = emb_mat[i][id_]
                except:
                    continue

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    train_model(args, model, dataloader_dict, optimizer, criterion, scheduler)