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
# from .model.transformer import PTransformer, Transformer
from .model.transformer.Models import PTransformer, Transformer
from .model.rnn import Encoder, Decoder, Seq2Seq
from .optimizer import Ralamb, WarmupLinearSchedule
from .training_module import model_training
from .utils import accuracy

def training(args):
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
            hj_train_indices = data_['hj_train_indices']
            hj_valid_indices = data_['hj_valid_indices']
            kr_train_indices = data_['kr_train_indices']
            kr_valid_indices = data_['kr_valid_indices']
            king_train_indices = data_['king_train_indices']
            king_valid_indices = data_['king_valid_indices']
            hj_word2id = data_['hj_word2id']
            kr_word2id = data_['kr_word2id']
            src_vocab_num = len(hj_word2id.keys())
            trg_vocab_num = len(kr_word2id.keys())
            del data_

    # with open(os.path.join(args.save_path_kr, 'kr_word2id.pkl'), 'rb') as f:
    #     kr_word2id = pickle.load(f)

    #===================================#
    #========DataLoader Setting=========#
    #===================================#

    dataset_dict = {
        'train': CustomDataset(hj_train_indices, kr_train_indices, king_train_indices,
                            min_len=args.min_len, src_max_len=args.src_max_len, 
                            trg_max_len=args.trg_max_len),
        'valid': CustomDataset(hj_valid_indices, kr_valid_indices, king_valid_indices,
                            min_len=args.min_len, src_max_len=args.src_max_len, 
                            trg_max_len=args.trg_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    }

    #====================================#
    #==========DWE Results Open==========#
    #====================================#

    if not args.src_baseline:
        with open(os.path.join(args.save_path, 'hj_emb_mat.pkl'), 'rb') as f:
            emb_mat_src = pickle.load(f)

    if not args.trg_baseline:
        with open(os.path.join(args.save_path_kr, 'emb_mat.pkl'), 'rb') as f:
            emb_mat_trg = pickle.load(f)

    #===================================#
    #===========Model Setting===========#
    #===================================#

    print("Build model")
    if 'transformer' in args.model_setting.lower():
        if args.model_setting == 'PTransformer':
            transformer_model_setting = PTransformer
        else:
            transformer_model_setting = Transformer
        model = transformer_model_setting(src_vocab_num, trg_vocab_num, 
                    pad_idx=args.pad_idx, bos_idx=args.bos_idx, eos_idx=args.eos_idx, 
                    src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                    d_model=args.d_model, d_embedding=args.d_embedding, 
                    n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                    dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                    num_encoder_layer=args.num_encoder_layer, num_decoder_layer=args.num_decoder_layer,
                    num_common_layer=args.num_common_layer,
                    src_baseline=args.src_baseline, trg_baseline=args.trg_baseline, 
                    share_qk=args.share_qk, swish_activation=args.swish_activation,
                    trg_emb_prj_weight_sharing=args.trg_emb_prj_weight_sharing,
                    emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing,
                    device=device)
    elif args.model_setting == 'rnn':
        encoder = Encoder(src_vocab_num, args.d_embedding, args.d_model, 
                        emb_mat_src, hj_word2id, n_layers=args.num_encoder_layer, 
                        pad_idx=args.pad_idx, dropout=args.dropout)
        decoder = Decoder(args.d_embedding, args.d_model, trg_vocab_num, n_layers=args.num_decoder_layer, 
                        pad_idx=args.pad_idx, dropout=args.dropout)
        model = Seq2Seq(encoder, decoder, device)
    else:
        raise Exception('Model error')

    # if args.resume:
    #     model_ner = NER_model(emb_mat=emb_mat, word2id=hj_word2id, pad_idx=args.pad_idx, bos_idx=args.bos_idx, eos_idx=args.eos_idx, max_len=args.max_len,
    #                     d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
    #                     dim_feedforward=args.dim_feedforward, n_layers=args.num_encoder_layer, dropout=args.dropout,
    #                     device=device)
    #     model_ner.load_state_dict(torch.load(os.path.join(args.save_path, 'ner_model_False.pt')))
    #     model.transformer_encoder.load_state_dict(model_ner.transformer_encoder.state_dict())
    #     for param in model.transformer_encoder.parameters():
    #         param.requires_grad = False
    # print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    optimizer = Ralamb(params=filter(lambda p: p.requires_grad, model.parameters()),
                       lr=args.lr, weight_decay=args.w_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=len(dataloader_dict['train'])*3, 
                                     t_total=len(dataloader_dict['train'])*args.num_epoch)
    # criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    model.to(device)

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    model_training(args, model, dataloader_dict, optimizer, scheduler, device)