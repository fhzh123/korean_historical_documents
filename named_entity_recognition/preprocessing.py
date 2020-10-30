# Import Module
import os
import time
import json
import pickle
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

# Import Custom Modules
from .utils import terminal_size, train_test_split, hj_encode_to_ids, ner_encode

def preprocessing(args):
    #===================================#
    #============Data Load==============#
    #===================================#

    print('#'*terminal_size())
    print('Total list making...')
    # 1) Path setting
    data_list = glob(os.path.join(args.ADJ_NER_path, '*/*.json'))
    data_list = sorted(data_list)[:-1] # 순종부록 제거

    total_string_list = list()
    total_ner_list = list()
    total_king_list = list()

    # 2) Total data making
    for data_path in tqdm(data_list):
        # 2-1) Load data
        with open(data_path, 'r') as f:
            data_ = json.load(f)
        data_string_list = list()
        data_ner_list = list()
        # 2-2) Extract string data by length
        for x in data_['string']:
            if len(x) <= args.max_len:
                data_string_list.append(x)
        # 2-3) Extract NER data by length
        for x in data_['NER']:
            if len(x) <= args.max_len:
                data_ner_list.append(x)
        # 2-4) Total data setting
        total_string_list.extend(data_string_list)
        total_ner_list.extend(data_ner_list)
        # 2-5) King list setting
        king_id = int(data_path.split('NER/')[1][:2]) - 1
        total_king_list.extend([king_id for _ in range(len(data_string_list))])

    #===================================#
    #============Data Split=============#
    #===================================#

    split_string_record, split_ner_record, split_king_record = train_test_split(
        total_string_list, total_ner_list, total_king_list, split_percent=args.data_split_per)

    print('Paired data num:')
    print(f"train: {len(split_string_record['train'])}")
    print(f"valid: {len(split_string_record['valid'])}")
    print(f"test: {len(split_string_record['test'])}")

    #====================================#
    #==========DWE Results Open==========#
    #====================================#

    with open(os.path.join(args.save_path, 'hj_word2id.pkl'), 'rb') as f:
        hj_word2id = pickle.load(f)['hanja_word2id']

    #===================================#
    #=======Hanja Pre-processing========#
    #===================================#

    print('Hanja sentence parsing...')
    start_time = time.time()

    # 1) Train data parsing (From utils.py)
    print('Train data start...')
    hj_parsed_indices_train = hj_encode_to_ids(split_string_record['train'], hj_word2id, args)

    # 2) Valid data parsing
    print('Valid data start...')
    hj_parsed_indices_valid = hj_encode_to_ids(split_string_record['valid'], hj_word2id, args)

    # 3) Test data parsing
    print('Test data start...')
    hj_parsed_indices_test = hj_encode_to_ids(split_string_record['test'], hj_word2id, args)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #========NER Pre-processing=========#
    #===================================#

    print('NER parsing...')
    start_time = time.time()

    # 1) Parsing sentence
    print('Train data start...')
    ner_indices_train = ner_encode(split_ner_record['train'])

    # 2) Test data parsing
    print('Valid data start...')
    ner_indices_valid = ner_encode(split_ner_record['valid'])

    # 3) Test data parsing
    print('Test data start...')
    ner_indices_test = ner_encode(split_ner_record['test'])

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Parsed sentence save setting...')
    start_time = time.time()

    with open(os.path.join(args.save_path, 'ner_processed.pkl'), 'wb') as f:
        pickle.dump({
            'hj_train_indices': hj_parsed_indices_train,
            'hj_valid_indices': hj_parsed_indices_valid,
            'hj_test_indices': hj_parsed_indices_test,
            'ner_train_indices': ner_indices_train,
            'ner_valid_indices': ner_indices_valid,
            'ner_test_indices': ner_indices_test,
            'king_train_indices': split_king_record['train'],
            'king_valid_indices': split_king_record['valid'],
            'king_test_indices': split_king_record['test'],
            'hj_word2id': hj_word2id,
            'hj_id2word': {v: k for k, v in hj_word2id.items()}
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')