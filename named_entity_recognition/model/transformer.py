# coding: utf-8
import numpy as np
from TorchCRF import CRF

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

class Transformer_model(nn.Module):
    def __init__(self, src_vocab_num, pad_idx=0, bos_idx=1, eos_idx=2, d_model=512, 
                 d_embedding=256, n_head=8, dim_feedforward=2048, n_layers=10, dropout=0.1, 
                 baseline=False, device=None):
        super(Transformer_model, self).__init__()

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.dropout = nn.Dropout(dropout)
        self.baseline = baseline

        # Source embedding part
        if baseline:
            self.src_embedding = nn.Embedding(src_vocab_num, d_model)
        else:
            self.src_embedding = TransformerEmbedding_with_bilinear(src_vocab_num, d_model, d_embedding)

        # Transformer
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward,
                activation='gelu', dropout=dropout) for i in range(n_layers)])
        # encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, 
        #                                             dropout=dropout, activation='gelu')
        # self.encoders = nn.TransformerEncoder(encoder_layers, n_layers)

        # Output Linear Part
        self.src_output_linear = nn.Linear(d_model, d_embedding)
        self.src_output_norm = nn.LayerNorm(d_embedding)
        self.src_output_linear2 = nn.Linear(d_embedding, 10)
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.src_output_linear.bias.data.zero_()
        self.src_output_linear.weight.data.uniform_(-initrange, initrange)
        self.src_output_linear2.bias.data.zero_()
        self.src_output_linear2.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, sequence, king_id, trg=None):
        if self.baseline:
            encoder_out = self.src_embedding(sequence).transpose(0, 1)
        else:
            encoder_out = self.src_embedding(sequence, king_id).transpose(0, 1)
        src_key_padding_mask = (sequence == self.pad_idx)

        # encoder_out = self.encoders(encoder_out, src_key_padding_mask=src_key_padding_mask)
        for i in range(len(self.encoders)):
            encoder_out = self.encoders[i](encoder_out, src_key_padding_mask=src_key_padding_mask)

        encoder_out = encoder_out.transpose(0, 1)
        encoder_out = self.src_output_norm(self.dropout(F.gelu(self.src_output_linear(encoder_out))))
        encoder_out = self.src_output_linear2(mish(encoder_out))
        # encoder_out = F.softmax(encoder_out, dim=2)
        return encoder_out

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, d_embedding, pad_idx=0):
        super().__init__()
        self.token_dict = dict()
        self.norm = nn.LayerNorm(d_model)
        self.linear_layer = nn.Bilinear(d_embedding, d_embedding, d_model)
        self.king_embedding = nn.Embedding(27, d_embedding)
    
    def forward(self, sequence, king_id):
        for i, king_ in enumerate(king_id):
            if i == 0:
                seq = self.token_dict[king_.item()](sequence[i]).unsqueeze(0)
            else:
                seq = torch.cat((seq, self.token_dict[king_.item()](sequence[i]).unsqueeze(0)), 0)
        x = self.linear_layer(seq, self.king_embedding(king_id).repeat(1, sequence.size(1), 1))
        return self.norm(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, dim_feedforward=2048, dropout=0.1, 
            activation="relu"):
        
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def mish(x): 
    return x * torch.tanh(F.softplus(x))