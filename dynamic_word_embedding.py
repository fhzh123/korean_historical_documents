# Import Modules
import os
import re
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import sentencepiece as spm

from glob import glob
from tqdm import tqdm
from gensim.corpora import Dictionary
from collections import deque, Counter

# Import PyTorch
import torch

# Import Custom Modules 
from dynamic_word_embedding.training import train_model # From 'https://github.com/llefebure/dynamic_bernoulli_embeddings'
from dynamic_word_embedding.utils import CustomError, dataframe_make

def main(args):

    # Path setting
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Data list setting
    AJD_data_path = glob(os.path.join(args.ADJ_data_path, '*.json'))
    AJD_data_path = sorted(AJD_data_path)[:-1] # 순종부록 제거

    # Preprocessing
    if args.Hanja_dwe:
        total_counter_hanja = Counter()
        comment_list_hanja = list()
    if args.Korean_dwe:
        if args.Korean_khaiii:
            total_counter_korean = Counter()
        else:
            for_parsing_list = list()
            record_korean_for_spm = list()
        comment_list_korean = list()
    king_list = list()
    king_index_list = list()

    # Data parsing & processing
    for ix, path in enumerate(tqdm(AJD_data_path)):
        with open(path, 'r') as f:
            record_list = json.load(f)
            king_list.append(path.split(' ')[-1][:2])
            king_index_list.append(ix)

            # Setting
            if args.Hanja_dwe:
                record_hanja = list()
            if args.Korean_dwe:
                record_korean = list()

            # Data appending
            for rc in record_list:
                if args.Hanja_dwe:
                    record_hanja.append(rc['hanja'])
                if args.Korean_dwe:
                    record_korean.append(rc['korean'])
                    record_korean_for_spm.append(rc['korean'])

            # Hanja processing
            if args.Hanja_dwe:
                record_hanja = ' '.join(record_hanja)
                hanja_ = re.sub(pattern='[a-zA-Z0-9]+', repl='', string=record_hanja)
                comment_list_hanja.append(hanja_)
                total_counter_hanja.update(hanja_)

            # Korean processing
            if args.Korean_dwe:
                if args.Korean_khaiii:
                    raise CustomError('Sorry; Will update soon!!')
                else:
                    for_parsing_list.append(record_korean)
                    total_record_by_king_str = ' '.join(record_korean)
                    comment_list_korean.append(total_record_by_king_str)
                
    # Hanja vocabulary setting
    if args.Hanja_dwe:
        hanja_vocab = list(total_counter_hanja.keys())
        # Will fix soon
        hanja_vocab.insert(0, '<unk>')
        hanja_vocab.insert(0, '</s>')
        hanja_vocab.insert(0, '<s>')
        hanja_vocab.insert(0, '<pad>')
        hanja_word2id = {w: i for i, w in enumerate(hanja_vocab)}

    # Korean parsing
    if args.Korean_dwe:
        if args.Korean_khaiii:
            raise CustomError('Sorry; Will update soon!!')
        else:
            # 1) Make Korean text to train vocab
            with open(f'{args.save_path}/korean.txt', 'w') as f:
                for korean in record_korean_for_spm:
                    f.write(f'{korean}\n')

            # 2) SentencePiece model training
            spm.SentencePieceProcessor()
            spm.SentencePieceTrainer.Train(
                f'--input={args.save_path}/korean.txt --model_prefix={args.save_path}/m_korean '
                f'--vocab_size={args.vocab_size} --character_coverage=0.995 --split_by_whitespace=true '
                f'--pad_id={args.pad_idx} --unk_id={args.unk_idx} --bos_id={args.bos_idx} --eos_id={args.eos_idx}')

            # 3) Korean vocabulary setting
            korean_vocab = list()
            with open(f'{args.save_path}/m_korean.vocab') as f:
                for line in f:
                    korean_vocab.append(line[:-1].split('\t')[0])
            korean_word2id = {w: i for i, w in enumerate(korean_vocab)}

            # 4) SentencePiece model load
            sp_kr = spm.SentencePieceProcessor()
            sp_kr.Load(f"{args.save_path}/m_korean.model")

            # 5) Korean parsing by SentencePiece model
            korean_pieces = list()
            for comment_ in tqdm(for_parsing_list):
                id_encoded_list = [sp_kr.EncodeAsPieces(korean)for korean in comment_]
                extended_id_list = list()
                for id_encoded in id_encoded_list:
                    extended_id_list.extend(id_encoded)
                korean_pieces.append(extended_id_list)

    # Dataset Setting (from utils.py)
    if args.Hanja_dwe:
        hanja_dataset = dataframe_make(king_list, comment_list_hanja, king_index_list)

    if args.Korean_dwe:
        korean_dataset = dataframe_make(king_list, korean_pieces, king_index_list)

    # Generate dictionary (will fix)
    if args.Hanja_dwe:
        hanja_dictionary = Dictionary(hanja_dataset.bow)
        hanja_dictionary.filter_extremes(no_below=15, no_above=1.)
        hanja_dictionary.compactify()
    
    if args.Korean_dwe:
        korean_dictionary = Dictionary(korean_dataset.bow)
        korean_dictionary.filter_extremes(no_below=15, no_above=1.)
        korean_dictionary.compactify()

    # Model training
    print('Hanja dynamic word embedding training...')
    if args.Hanja_dwe:
        hanja_model, hanja_loss_history = train_model(hanja_dataset, hanja_word2id, validation=None, m=args.minibatch_iteration,
                                                      num_epochs=args.num_epochs, notebook=False)

    print('Korean dynamic word embedding training...')
    if args.Korean_dwe:
        korean_model, korean_loss_history = train_model(korean_dataset, korean_word2id, validation=None, m=args.minibatch_iteration,
                                                        num_epochs=args.num_epochs, notebook=False)

    # Model saving
    print('Saving...')
    if args.Hanja_dwe:
        torch.save(hanja_model, os.path.join(args.save_path, 'dwe_hj_model.pt'))
    if args.Korean_dwe:
        torch.save(korean_model, os.path.join(args.save_path, 'dwe_kr_model.pt'))

    # Embedding vector saving
    if args.Hanja_dwe:
        with open(os.path.join(args.save_path, 'hj_emb_mat.pkl'), 'wb') as f:
            pickle.dump(hanja_model.get_embeddings(), f)
    if args.Korean_dwe:
        with open(os.path.join(args.save_path, 'kr_emb_mat.pkl'), 'wb') as f:
            pickle.dump(korean_model.get_embeddings(), f)

    # Pre-processed results saving
    if args.Hanja_dwe:
        with open(os.path.join(args.save_path, 'hj_word2id.pkl'), 'wb') as f:
            pickle.dump({
                'hanja_word2id': hanja_word2id,
                'total_counter_hanja': total_counter_hanja
            }, f)
    if args.Korean_dwe:
        with open(os.path.join(args.save_path, 'kr_word2id.pkl'), 'wb') as f:
            pickle.dump(korean_word2id, f)

    # Loss saving
    if args.Hanja_dwe:
        hanja_loss_history.to_csv(os.path.join(args.save_path, 'hanja_loss_history.csv'), index=False)
    if args.Korean_dwe:
        korean_loss_history.to_csv(os.path.join(args.save_path, 'korean_loss_history.csv'), index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DWE argparser')
    # Language select
    parser.add_argument('--Hanja_dwe', type=bool, default=True, help='Hanja processing')
    parser.add_argument('--Korean_dwe', type=bool, default=True, help='Korean processing')
    parser.add_argument('--Korean_khaiii', type=bool, default=False)
    # Path setting
    parser.add_argument('--ADJ_data_path', type=str, default='./data', 
                        help='Data path setting')
    parser.add_argument('--save_path', type=str, default='./preprocessing',
                        help='Save path setting')
    # Training setting
    parser.add_argument('--num_epochs', type=int, default=5, help='The number of epoch')
    parser.add_argument('--minibatch_iteration', type=int, default=300, help='Mini-batch iteration')
    parser.add_argument('--pad_idx', default=0, type=int, help='Padding index')
    parser.add_argument('--bos_idx', default=1, type=int, help='Start token index')
    parser.add_argument('--eos_idx', default=2, type=int, help='End token index')
    parser.add_argument('--unk_idx', default=3, type=int, help='Unknown token index')
    parser.add_argument('--vocab_size', default=24000, type=int, help='Korean vocabulary size')
    args = parser.parse_args()

    total_start_time = time.time()
    main(args)
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')