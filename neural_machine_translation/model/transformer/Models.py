''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from .Layers import EncoderLayer, DecoderLayer


__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.embedding_norm = nn.LayerNorm(d_word_vec, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.embedding_norm(self.position_enc(self.src_word_emb(src_seq)))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.embedding_norm = nn.LayerNorm(d_word_vec, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.embedding_norm(self.position_enc(self.trg_word_emb(trg_seq)))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        dec_output = self.layer_norm(dec_output)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, src_vocab_num, trg_vocab_num, pad_idx=0, bos_idx=1, eos_idx=2,
            src_max_len=300, trg_max_len=360, d_model=512, d_embedding=512,
            n_head=8, d_k=64, d_v=64, dim_feedforward=2048, dropout=0.1,
            num_encoder_layer=8, num_decoder_layer=8, num_common_layer=0,
            src_baseline=False, trg_baseline=False, share_qk=False, swish_activation=False,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=False,
            device=None):

        super().__init__()

        self.pad_idx = pad_idx
        self.bos_dix = bos_idx
        self.eos_idx = eos_idx

        self.encoder = Encoder(
            n_src_vocab=src_vocab_num, n_position=src_max_len,
            d_word_vec=d_embedding, d_model=d_model, d_inner=dim_feedforward,
            n_layers=num_encoder_layer, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_trg_vocab=trg_vocab_num, n_position=trg_max_len,
            d_word_vec=d_embedding, d_model=d_model, d_inner=dim_feedforward,
            n_layers=num_decoder_layer, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=pad_idx, dropout=dropout)

        self.trg_word_prj = nn.Linear(d_model, trg_vocab_num, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_embedding, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))


class PTransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, src_vocab_num, trg_vocab_num, pad_idx=0, bos_idx=1, eos_idx=2,
            src_max_len=300, trg_max_len=360, d_model=512, d_embedding=512,
            n_head=8, d_k=64, d_v=64, dim_feedforward=2048, dropout=0.1,
            num_encoder_layer=8, num_decoder_layer=8, num_common_layer=8,
            src_baseline=False, trg_baseline=False, share_qk=False, swish_activation=False,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=False,
            device=None):

        super().__init__()

        self.pad_idx = pad_idx
        self.bos_dix = bos_idx
        self.eos_idx = eos_idx

        self.num_common_layer = num_common_layer
        self.n_encoder_nonparallel = num_encoder_layer - num_common_layer
        
        self.src_word_emb = nn.Embedding(src_vocab_num, d_embedding, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_embedding, n_position=src_max_len)
        self.src_embedding_norm = nn.LayerNorm(d_embedding, eps=1e-6)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, dim_feedforward, n_head, d_k, d_v, dropout=dropout, 
                share_qk=share_qk, swish_activation=swish_activation) for _ in range(num_encoder_layer)])
        self.encoder_norms = nn.ModuleList([
            nn.LayerNorm(d_model, eps=1e-6) for _ in range(self.num_common_layer)])

        self.trg_word_emb = nn.Embedding(trg_vocab_num, d_embedding, padding_idx=pad_idx)
        self.trg_embedding_norm = nn.LayerNorm(d_embedding, eps=1e-6)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, dim_feedforward, n_head, d_k, d_v, dropout=dropout)
            for _ in range(num_decoder_layer)])
        self.decoder_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.trg_word_prj = nn.Linear(d_model, trg_vocab_num, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_embedding, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.src_word_emb.weight = self.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, self.pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.pad_idx) & get_subsequent_mask(trg_seq)
        enc_output = self.src_embedding_norm(self.position_enc(self.src_word_emb(src_seq)))
        dec_output = self.trg_embedding_norm(self.position_enc(self.trg_word_emb(trg_seq)))

        for enc_layer in self.encoder_layers[:self.n_encoder_nonparallel+1]:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
        
        for enc_layer, enc_norm, dec_layer in zip(
                self.encoder_layers[self.n_encoder_nonparallel+1:],
                self.encoder_norms[:-1],
                self.decoder_layers[:self.num_common_layer-1]):
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_norm(enc_output), slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            
        enc_output = self.encoder_norms[-1](enc_output)
        for dec_layer in self.decoder_layers[self.num_common_layer-1:]:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            
        dec_output = self.decoder_layer_norm(dec_output)
        seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale
        return seq_logit.view(-1, seq_logit.size(2))