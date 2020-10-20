import torch
import torch.nn as nn

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, d_embedding, pad_idx=0):
        super().__init__()
        self.token_dict = dict()
        self.norm = nn.LayerNorm(d_model)
        self.linear_layer = nn.Bilinear(d_embedding, d_embedding, d_model)
        self.king_embedding = nn.Embedding(27, d_embedding)
        
        self.position = PositionalEmbedding(d_model=embed_size, max_len=max_len)

    def forward(self, sequence, king_id):
        for i, king_ in enumerate(king_id):
            if i == 0:
                seq = self.token_dict[king_.item()](sequence[i]).unsqueeze(0)
            else:
                seq = torch.cat((seq, self.token_dict[king_.item()](sequence[i]).unsqueeze(0)), 0)
        x = self.linear_layer(seq, self.king_embedding(king_id).repeat(1, sequence.size(1), 1))
        return self.norm(x)