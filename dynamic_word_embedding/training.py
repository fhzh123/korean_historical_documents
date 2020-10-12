# Import Modules
import os
import pickle
import argparse
import numpy as np
import pandas as pd

# Import PyTorch
import torch

# Import Custom Modules 
from dynamic_word_embedding.module import train_model # From 'https://github.com/llefebure/dynamic_bernoulli_embeddings'

def training(args):
    # Setting
    if args.Hanja_dwe:
        hanja_dataset = pd.read_csv(os.path.join(args.save_path, 'hanja_dataset.csv'))
        with open(os.path.join(args.save_path, 'hj_word2id.pkl'), 'rb') as f:
            hanja_word2id = pickle.load(f)['hanja_word2id']

    if args.Korean_dwe:
        korean_dataset = pd.read_csv(os.path.join(args.save_path, 'korean_dataset.csv'))
        with open(os.path.join(args.save_path, 'kr_word2id.pkl'), 'rb') as f:
            korean_word2id = pickle.load(f)['korean_word2id']

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

    # Loss saving
    if args.Hanja_dwe:
        hanja_loss_history.to_csv(os.path.join(args.save_path, 'hanja_loss_history.csv'), index=False)
    if args.Korean_dwe:
        korean_loss_history.to_csv(os.path.join(args.save_path, 'korean_loss_history.csv'), index=False)