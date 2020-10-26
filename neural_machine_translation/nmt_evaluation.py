# Import Module
import os
import math
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import sentencepiece as spm
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
from .model.transformer import Transformer
from .model.rnn import Encoder, Decoder, Seq2Seq
from .utils import accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    print('Data Load & Setting!')
    if args.dataset == 'total_data':
        with open(os.path.join(args.save_path, 'preprocessed_data.json'), 'r') as f:
            data_ = pickle.load(f)
            hj_test_indices = data_['test_hanja_indices']
            kr_test_indices = data_['test_korean_indices']
            king_test_indices = data_['king_ids_hk_test']
            hj_word2id = data_['hanja_word2id']
            kr_word2id = data_['korean_word2id']
            src_vocab_num = len(src_word2id.keys())
            trg_vocab_num = len(trg_word2id.keys())
            del data_

    elif args.dataset == 'normal_data':
        with open(os.path.join(args.save_path, 'nmt_processed.pkl'), 'rb') as f:
            data_ = pickle.load(f)
            hj_test_indices = data_['hj_test_indices']
            kr_test_indices = data_['kr_test_indices']
            king_test_indices = data_['king_test_indices']
            hj_word2id = data_['hj_word2id']
            hj_id2word = data_['hj_id2word']
            kr_word2id = data_['kr_word2id']
            kr_id2word = data_['kr_id2word']
            src_vocab_num = len(hj_word2id.keys())
            trg_vocab_num = len(kr_word2id.keys())
            del data_

    #===================================#
    #========DataLoader Setting=========#
    #===================================#

    testDataset = CustomDataset(hj_test_indices, kr_test_indices, king_test_indices,
                                min_len=args.min_len, src_max_len=args.src_max_len, 
                                trg_max_len=args.trg_max_len)
    testDataloader = DataLoader(testDataset, collate_fn=PadCollate(), drop_last=True,
                                batch_size=args.batch_size, shuffle=True, 
                                num_workers=4, pin_memory=True)

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
    if args.model_setting == 'transformer':
        model = Transformer(src_vocab_num, trg_vocab_num, 
                    pad_idx=args.pad_idx, bos_idx=args.bos_idx, eos_idx=args.eos_idx, 
                    src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                    d_model=args.d_model, d_embedding=args.d_embedding, 
                    n_head=args.n_head, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                    num_encoder_layer=args.num_encoder_layer, num_decoder_layer=args.num_decoder_layer,
                    src_baseline=args.src_baseline, trg_baseline=args.trg_baseline, device=device)
    elif args.model_setting == 'rnn':
        encoder = Encoder(src_vocab_num, args.d_embedding, args.d_model, 
                        emb_mat_src, hj_word2id, n_layers=args.num_encoder_layer, 
                        pad_idx=args.pad_idx, dropout=args.dropout)
        decoder = Decoder(args.d_embedding, args.d_model, trg_vocab_num, n_layers=args.num_decoder_layer, 
                        pad_idx=args.pad_idx, dropout=args.dropout)
        model = Seq2Seq(encoder, decoder, device)
    else:
        raise Exception('Model error')

    model.load_state_dict(torch.load('./preprocessing/nmt_model_transformer_testing.pt'))
    model.to(device)
    model.eval()

    #===================================#
    #==============Testing==============#
    #===================================#

    real_list = list()
    pred_list = list()

    for src, trg, king_id in tqdm(dataloader_dict['test']):
        # Sourcen, Target sentence setting
        label_sequences = trg.to(device, non_blocking=True)
        input_sequences = src.to(device, non_blocking=True)
        king_id = king_id.to(device, non_blocking=True)

        non_pad = label_sequences != args.pad_idx
        trg_sequences_target = label_sequences[non_pad].contiguous().view(-1)

        if args.model_setting == 'transformer':
            # Target Masking
            tgt_mask = model.generate_square_subsequent_mask(label_sequences.size(1))
            tgt_mask = tgt_mask.to(device, non_blocking=True)

        # Model / Calculate loss
        with torch.no_grad():
            if args.model_setting == 'transformer':
                predicted = model(input_sequences, label_sequences, king_id, tgt_mask, non_pad)

        pred_list.append(predicted.topk(1,1,True,True)[1].squeeze(1).tolist())
        real_list.append(trg_sequences_target.tolist())

    spm_ = spm.SentencePieceProcessor()
    spm_.Load(os.path.join(args.save_path, 'm_korean.model'))

    with open(os.path.join(args.save_path, 'pred5.txt'), 'w') as f:
        for pred in pred_list:
            f.write(spm_.decode_ids(pred))
            f.write('\n')

    with open(os.path.join(args.save_path, 'real5.txt'), 'w') as f:
        for real in real_list:
            f.write(spm_.decode_ids(real))
            f.write('\n')

    print('''type 'nlg-eval --hypothesis=save3/pred5.txt --references=save3/real5.txt' in terminal''')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='NMT argparser')
    parser.add_argument('--save_path', default='./save3', 
                        type=str, help='path of data pickle file (train)')
    parser.add_argument('--resume', action='store_true',
                        help='If not store, then training from scratch')
    parser.add_argument('--baseline', action='store_true',
                        help='If not store, then training from Dynamic Word Embedding')
    parser.add_argument('--save_path_kr', default='./save_korean2', 
                        type=str, help='path of data pickle file (train)')
    parser.add_argument('--model_setting', default='transformer', choices=['transformer', 'rnn'],
                        type=str, help='Model Selection; transformer vs rnn')

    parser.add_argument('--pad_idx', default=0, type=int, help='pad index')
    parser.add_argument('--bos_idx', default=1, type=int, help='index of bos token')
    parser.add_argument('--eos_idx', default=2, type=int, help='index of eos token')
    parser.add_argument('--unk_idx', default=3, type=int, help='index of unk token')

    parser.add_argument('--min_len', type=int, default=4, help='Minimum Length of Sentences; Default is 4')
    parser.add_argument('--max_len', type=int, default=500, help='Max Length of Source Sentence; Default is 150')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size; Default is 48')
    parser.add_argument('--d_model', type=int, default=512, help='Hidden State Vector Dimension; Default is 512')
    parser.add_argument('--d_embedding', type=int, default=256, help='Embedding Vector Dimension; Default is 256')
    parser.add_argument('--n_head', type=int, default=8, help='Multihead Count; Default is 256')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Embedding Vector Dimension; Default is 512')
    parser.add_argument('--num_encoder_layer', default=8, type=int, help='number of encoder layer')
    parser.add_argument('--num_decoder_layer', default=8, type=int, help='number of decoder layer')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout Ratio; Default is 0.5')

    args = parser.parse_args()
    main(args)