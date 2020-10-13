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
from .model import NER_model
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
        'valid': CustomDataset(hj_test_indices, ner_test_indices, king_test_indices,
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
    model = NER_model(src_vocab_num=src_vocab_num, pad_idx=args.pad_idx, bos_idx=args.bos_idx, 
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NER argparser')
    parser.add_argument('--save_path', default='./save', 
                        type=str, help='path of data pickle file (train)')
    parser.add_argument('--pad_idx', default=0, type=int, help='pad index')
    parser.add_argument('--bos_idx', default=1, type=int, help='index of bos token')
    parser.add_argument('--eos_idx', default=2, type=int, help='index of eos token')
    parser.add_argument('--unk_idx', default=3, type=int, help='index of unk token')

    parser.add_argument('--min_len', type=int, default=4, help='Minimum Length of Sentences; Default is 4')
    parser.add_argument('--max_len', type=int, default=200, help='Max Length of Source Sentence; Default is 150')

    parser.add_argument('--num_epoch', type=int, default=10, help='Epoch count; Default is 10')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size; Default is 48')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate; Default is 5e-4')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay; Default is 0.5')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='Learning rate decay step; Default is 5')
    parser.add_argument('--grad_clip', type=int, default=5, help='Set gradient clipping; Default is 5')
    parser.add_argument('--w_decay', type=float, default=1e-6, help='Weight decay; Default is 1e-6')

    parser.add_argument('--d_model', type=int, default=768, help='Hidden State Vector Dimension; Default is 512')
    parser.add_argument('--d_embedding', type=int, default=256, help='Embedding Vector Dimension; Default is 256')
    parser.add_argument('--n_head', type=int, default=8, help='Multihead Count; Default is 256')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Embedding Vector Dimension; Default is 512')
    parser.add_argument('--n_layers', type=int, default=8, help='Model layers; Default is 8')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout Ratio; Default is 0.5')
    parser.add_argument('--baseline', action='store_true', help='Do not use bilinear embedding')

    parser.add_argument('--print_freq', type=int, default=300, help='Print train loss frequency; Default is 100')
    args = parser.parse_args()

    main(args)