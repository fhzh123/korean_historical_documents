# Import PyTorch
import torch
import torch.nn.functional as F
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, emb_mat, src_word2id,
                 n_layers=1, pad_idx=0, dropout=0.0, embedding_dropout=0.0):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.sru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=pad_idx)
        # self.embedding = TransformerEmbedding_bilinear(len(src_word2id.keys()), 
        #                 self.hidden_size, self.embed_size, emb_mat, src_word2id)
        
    def forward(self, src, king_id, hidden=None, cell=None):
        # Source sentence embedding
        embeddings = self.embedding(src, king_id)  # (max_caption_length, batch_size, embed_dim)
        embedded = F.dropout(embeddings, p=self.embedding_dropout, inplace=True) # (max_caption_length, batch_size, embed_dim)
        # Bidirectional SRU
        outputs, hidden = self.sru(embedded, hidden)
        # sum bidirectional outputs
        outputs = torch.tanh(self.linear(outputs)) # (max_caption_length, batch_size, embed_dim)
        return outputs, hidden