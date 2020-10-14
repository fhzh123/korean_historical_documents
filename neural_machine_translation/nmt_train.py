# Import Module
import os
import math
import time
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
from translation.dataset import CustomDataset, PadCollate
from translation.model import Transformer
from translation.optimizer import Ralamb, WarmupLinearSchedule
from translation.rnn_model import Encoder, Decoder, Seq2Seq
from named_entity_recognition.model import NER_model
from utils import accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    print('Data Load & Setting!')
    with open(os.path.join(args.save_path, 'nmt_processed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        hj_train_indices = data_['hj_train_indices']
        hj_valid_indices = data_['hj_valid_indices']
        hj_test_indices = data_['hj_test_indices']
        kr_train_indices = data_['kr_train_indices']
        kr_valid_indices = data_['kr_valid_indices']
        kr_test_indices = data_['kr_test_indices']
        king_train_indices = data_['king_train_indices']
        king_valid_indices = data_['king_valid_indices']
        king_test_indices = data_['king_test_indices']
        hj_word2id = data_['hj_word2id']
        hj_id2word = data_['hj_id2word']
        src_vocab_num = len(hj_word2id.keys())
        # trg_vocab_num = len(kr_word2id.keys())
        trg_vocab_num = 24000
        del data_

    with open(os.path.join(args.save_path_kr, 'kr_word2id.pkl'), 'rb') as f:
        kr_word2id = pickle.load(f)

    #===================================#
    #========DataLoader Setting=========#
    #===================================#

    dataset_dict = {
        'train': CustomDataset(hj_train_indices, kr_train_indices, king_train_indices,
                            min_len=args.min_len, max_len=args.max_len),
        'valid': CustomDataset(hj_valid_indices, kr_test_indices, king_test_indices,
                            min_len=args.min_len, max_len=args.max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True)
    }

    #====================================#
    #==========DWE Results Open==========#
    #====================================#

    with open(os.path.join(args.save_path, 'emb_mat.pkl'), 'rb') as f:
        emb_mat_src = pickle.load(f)

    with open(os.path.join(args.save_path_kr, 'emb_mat.pkl'), 'rb') as f:
        emb_mat_trg = pickle.load(f)

    #===================================#
    #===========Model Setting===========#
    #===================================#

    print("Build model")
    if args.model_setting == 'transformer':
        model = Transformer(emb_mat_src, emb_mat_trg,
                    hj_word2id, kr_word2id, src_vocab_num, trg_vocab_num, 
                    pad_idx=args.pad_idx, bos_idx=args.bos_idx, 
                    eos_idx=args.eos_idx, max_len=args.max_len,
                    d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head, 
                    dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                    num_encoder_layer=args.num_encoder_layer, num_decoder_layer=args.num_decoder_layer,
                    baseline=args.baseline, device=device)
    elif args.model_setting == 'rnn':
        encoder = Encoder(src_vocab_num, args.d_embedding, args.d_model, 
                        emb_mat_src, hj_word2id, n_layers=args.num_encoder_layer, 
                        pad_idx=args.pad_idx, dropout=args.dropout)
        decoder = Decoder(args.d_embedding, args.d_model, trg_vocab_num, n_layers=args.num_decoder_layer, 
                        pad_idx=args.pad_idx, dropout=args.dropout)
        model = Seq2Seq(encoder, decoder, device)
    else:
        raise Exception('Model error')

    if args.resume:
        model_ner = NER_model(emb_mat=emb_mat, word2id=hj_word2id, pad_idx=args.pad_idx, bos_idx=args.bos_idx, eos_idx=args.eos_idx, max_len=args.max_len,
                        d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
                        dim_feedforward=args.dim_feedforward, n_layers=args.num_encoder_layer, dropout=args.dropout,
                        device=device)
        model_ner.load_state_dict(torch.load(os.path.join(args.save_path, 'ner_model_False.pt')))
        model.transformer_encoder.load_state_dict(model_ner.transformer_encoder.state_dict())
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=len(dataloader_dict['train'])*3, 
                                     t_total=len(dataloader_dict['train'])*args.num_epoch)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)
    model.to(device)

    best_val_loss = None
    total_train_loss_list = list()
    total_test_loss_list = list()
    freq = 0
    for e in range(args.num_epoch):
        start_time_e = time.time()
        print(f'Model Fitting: [{e+1}/{args.num_epoch}]')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()
                val_loss = 0
                val_top1_acc = 0
                val_top5_acc = 0
                val_top10_acc = 0
            for i, (src, trg, king_id) in enumerate(dataloader_dict[phase]):
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
                    tgt_mask = tgt_mask.transpose(0, 1)

                # Optimizer setting
                optimizer.zero_grad()

                # Model / Calculate loss
                with torch.set_grad_enabled(phase == 'train'):
                    if args.model_setting == 'transformer':
                        predicted = model(input_sequences, label_sequences, king_id, tgt_mask, non_pad)
                        loss = criterion(predicted, trg_sequences_target)
                    if args.model_setting == 'rnn':
                        teacher_forcing_ratio_ = 0.5
                        input_sequences = input_sequences.transpose(0, 1)
                        label_sequences = label_sequences.transpose(0, 1)
                        predicted = model(input_sequences, label_sequences, king_id, 
                                        teacher_forcing_ratio=teacher_forcing_ratio_)
                        predicted = predicted.view(-1, trg_vocab_num)
                        trg_sequences_target = label_sequences.contiguous().view(-1)
                        loss = criterion(predicted, trg_sequences_target)
                    if phase == 'valid':
                        val_loss += loss.item()
                        top1_acc, top5_acc, top10_acc = accuracy(predicted, 
                                                                 trg_sequences_target, 
                                                                 topk=(1,5,10))
                        val_top1_acc += top1_acc.item()
                        val_top5_acc += top5_acc.item()
                        val_top10_acc += top10_acc.item()
                # If phase train, then backward loss and step optimizer and scheduler
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    total_train_loss_list.append(loss.item())

                    # Print loss value only training
                    freq += 1
                    if freq == args.print_freq:
                        total_loss = loss.item()
                        top1_acc, top5_acc, top10_acc = accuracy(predicted, 
                                                                 trg_sequences_target, 
                                                                 topk=(1,5,10))
                        print("[Epoch:%d][%d/%d] train_loss:%5.3f | top1_acc:%5.2f | top5_acc:%5.2f | spend_time:%5.2fmin"
                                % (e+1, i, len(dataloader_dict['train']), total_loss, top1_acc, top5_acc, (time.time() - start_time_e) / 60))
                        freq = 0

            # Finishing iteration
            if phase == 'valid':
                val_loss /= len(dataloader_dict['valid'])
                val_top1_acc /= len(dataloader_dict['valid'])
                val_top5_acc /= len(dataloader_dict['valid'])
                val_top10_acc /= len(dataloader_dict['valid'])
                total_test_loss_list.append(val_loss)
                print("[Epoch:%d] val_loss:%5.3f | top1_acc:%5.2f | top5_acc:%5.2f | top10_acc:%5.2f | spend_time:%5.2fmin"
                        % (e+1, val_loss, val_top1_acc, val_top5_acc, val_top10_acc, (time.time() - start_time_e) / 60))
                if not best_val_loss or val_loss > best_val_loss:
                    print("[!] saving model...")
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    torch.save(model.state_dict(), 
                               os.path.join(args.save_path, f'nmt_model_{args.model_setting}_{args.resume}_{args.baseline}_testing2.pt'))
                    best_val_loss = val_loss

        # Learning rate scheduler setting
        # scheduler.step()

    pd.DataFrame(total_train_loss_list).to_csv(os.path.join(args.save_path, f'train_loss_{args.baseline}_{args.model_setting}_{args.resume}.csv'), index=False)
    pd.DataFrame(total_test_loss_list).to_csv(os.path.join(args.save_path, f'test_loss_{args.baseline}_{args.model_setting}_{args.resume}.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NMT argparser')
    parser.add_argument('--save_path', default='./save', 
                        type=str, help='path of data pickle file (train)')
    parser.add_argument('--save_path_kr', default='./save_korean2', 
                        type=str, help='path of data pickle file (train)')
    parser.add_argument('--resume', action='store_true',
                        help='If not store, then training from scratch')
    parser.add_argument('--baseline', action='store_true',
                        help='If not store, then training from Dynamic Word Embedding')
    parser.add_argument('--model_setting', default='transformer', choices=['transformer', 'rnn'],
                        type=str, help='Model Selection; transformer vs rnn')
    parser.add_argument('--pad_idx', default=0, type=int, help='pad index')
    parser.add_argument('--bos_idx', default=1, type=int, help='index of bos token')
    parser.add_argument('--eos_idx', default=2, type=int, help='index of eos token')
    parser.add_argument('--unk_idx', default=3, type=int, help='index of unk token')

    parser.add_argument('--min_len', type=int, default=4, help='Minimum Length of Sentences; Default is 4')
    parser.add_argument('--max_len', type=int, default=500, help='Max Length of Source Sentence; Default is 150')

    parser.add_argument('--num_epoch', type=int, default=10, help='Epoch count; Default is 10')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size; Default is 48')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate; Default is 5e-5')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay; Default is 0.5')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='Learning rate decay step; Default is 5')
    parser.add_argument('--grad_clip', type=int, default=5, help='Set gradient clipping; Default is 5')
    parser.add_argument('--w_decay', type=float, default=1e-6, help='Weight decay; Default is 1e-6')

    parser.add_argument('--d_model', type=int, default=512, help='Hidden State Vector Dimension; Default is 512')
    parser.add_argument('--d_embedding', type=int, default=256, help='Embedding Vector Dimension; Default is 256')
    parser.add_argument('--n_head', type=int, default=8, help='Multihead Count; Default is 256')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Embedding Vector Dimension; Default is 512')
    parser.add_argument('--num_encoder_layer', default=8, type=int, help='number of encoder layer')
    parser.add_argument('--num_decoder_layer', default=8, type=int, help='number of decoder layer')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout Ratio; Default is 0.5')

    parser.add_argument('--print_freq', type=int, default=300, help='Print train loss frequency; Default is 100')
    args = parser.parse_args()
    main(args)