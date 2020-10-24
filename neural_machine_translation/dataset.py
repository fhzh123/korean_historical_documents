# Import Module
import sentencepiece as spm
from itertools import chain
from random import random, randrange

import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, src_list, trg_list, king_list, min_len=4, src_max_len=300, trg_max_len=360):
        data = list()
        for h, k, king_id in zip(src_list, trg_list, king_list):
            if min_len <= len(h) <= src_max_len and min_len <= len(k) <= trg_max_len:
                data.append((h, k, king_id))
        
        self.data = data
        self.num_data = len(self.data)
        
    def __getitem__(self, index):
        hanja, korean, king_id = self.data[index]
        return hanja, korean, king_id
    
    def __len__(self):
        return self.num_data

class PadCollate:
    def __init__(self, pad_index=0, dim=0):
        self.dim = dim
        self.pad_index = pad_index

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)

        def pack_sentence(sentences):
            sentences_len = max(map(lambda x: len(x), sentences))
            sentences = [pad_tensor(torch.LongTensor(seq), sentences_len, self.dim) for seq in sentences]
            sentences = torch.cat(sentences)
            sentences = sentences.view(-1, sentences_len)
            return sentences

        input_sentences, output_sentences, king_ids = zip(*batch)
        king_ids = torch.LongTensor(king_ids).view(-1, 1)
        return pack_sentence(input_sentences), pack_sentence(output_sentences), king_ids

    def __call__(self, batch):
        return self.pad_collate(batch)