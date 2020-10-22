# Import Modules
import os
import json
import time
import pickle
import argparse
import numpy as np
import sentencepiece as spm
from glob import glob
from tqdm import tqdm
from collections import Counter

# Import Custom Modules
from .utils import terminal_size, train_test_split, hj_encode_to_ids

def preprocessing(args):
    #===================================#
    #============Data Load==============#
    #===================================#

    print('Total list making...')
    # 1) Path setting
    data_list = glob(os.path.join(args.ADJ_data_path, '*.json'))
    data_list = sorted(data_list)[:-1] # 순종부록 제거

    total_src_list = list()
    total_trg_list = list()
    total_king_list = list()

    # 2) Total data making
    for data_path in tqdm(data_list):
        # 2-1) Load data
        with open(data_path, 'r') as f:
            data_ = json.load(f)
        data_src_list = list()
        data_trg_list = list()
        # 2-2) Extract string data by length
        for x in data_:
            data_src_list.append(x['hanja'])
            data_trg_list.append(x['korean'])
        # 2-3) Total data setting
        total_src_list.extend(data_src_list)
        total_trg_list.extend(data_trg_list)
        # 2-4) King list setting
        king_id = int(data_path.split('data/')[1][:2]) - 1 # Start from 0
        total_king_list.extend([king_id for _ in range(len(data_src_list))])

    #===================================#
    #============Data Split=============#
    #===================================#

    split_src_record, split_trg_record, split_king_record = train_test_split(
        total_src_list, total_trg_list, total_king_list, split_percent=args.data_split_per)

    print('Paired data num:')
    print(f"train: {len(split_src_record['train'])}")
    print(f"valid: {len(split_src_record['valid'])}")
    print(f"test: {len(split_src_record['test'])}")

    #====================================#
    #==========DWE Results Open==========#
    #====================================#

    with open(os.path.join(args.save_path, 'hj_word2id.pkl'), 'rb') as f:
        hj_word2id = pickle.load(f)['hanja_word2id']

    #===================================#
    #=======Hanja Pre-processing========#
    #===================================#

    # 1) Hanja sentence parsing setting
    print('Hanja sentence parsing...')
    start_time = time.time()
    hj_parsed_indices_train = list()
    hj_parsed_indices_valid = list()
    hj_parsed_indices_test = list()

    # 2) Parsing sentence

    word_counter = Counter()
    hanja_word2id = ['<pad>', '<s>', '</s>', '<unk>']
    # Hanja Counter
    for sentence in split_src_record['train']:
        for word in sentence:
            word_counter.update(word)

    hanja_word2id.extend([w for w, freq in word_counter.items() if freq >= 10])
    hj_word2id = {w: i for i, w in enumerate(hanja_word2id)}

    # 2-1) Train data parsing
    # 1) Train data parsing (From utils.py)

    print('Train data start...')
    hj_parsed_indices_train = hj_encode_to_ids(split_src_record['train'], hj_word2id, args)

    # 2) Valid data parsing
    print('Valid data start...')
    hj_parsed_indices_valid = hj_encode_to_ids(split_src_record['valid'], hj_word2id, args)

    # 3) Test data parsing
    print('Test data start...')
    hj_parsed_indices_test = hj_encode_to_ids(split_src_record['test'], hj_word2id, args)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #=======Korean Pre-processing=======#
    #===================================#

    # 1) Make Korean text to train vocab
    with open(f'{args.save_path}/korean.txt', 'w') as f:
        for korean in split_trg_record['train']:
            f.write(f'{korean}\n')

    # 2) SentencePiece model training
    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.save_path}/korean.txt --model_prefix={args.save_path}/m_korean --model_type=bpe '
        f'--vocab_size={args.kr_vocab_size} --character_coverage=0.995 --split_by_whitespace=true '
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
    train_korean_indices = [[args.bos_idx] + sp_kr.EncodeAsIds(korean) + [args.eos_idx] for korean in split_trg_record['train']]
    valid_korean_indices = [[args.bos_idx] + sp_kr.EncodeAsIds(korean) + [args.eos_idx] for korean in split_trg_record['valid']]
    test_korean_indices = [[args.bos_idx] + sp_kr.EncodeAsIds(korean) + [args.eos_idx] for korean in split_trg_record['test']]

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Parsed sentence save setting...')
    start_time = time.time()

    with open(os.path.join(args.save_path, 'nmt_processed.pkl'), 'wb') as f:
        pickle.dump({
            'hj_train_indices': hj_parsed_indices_train,
            'hj_valid_indices': hj_parsed_indices_valid,
            'hj_test_indices': hj_parsed_indices_test,
            'kr_train_indices': train_korean_indices,
            'kr_valid_indices': valid_korean_indices,
            'kr_test_indices': test_korean_indices,
            'king_train_indices': split_king_record['train'],
            'king_valid_indices': split_king_record['valid'],
            'king_test_indices': split_king_record['test'],
            'hj_word2id': hj_word2id,
            'kr_word2id': korean_word2id,
            'hj_id2word': {v: k for k, v in hj_word2id.items()},
            'kr_id2word': {v: k for k, v in korean_word2id.items()}
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')