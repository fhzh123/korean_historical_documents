# Import modules
import math

# Import PyTorch
import torch
import torch.nn.functional as F
from torch import nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size, 1))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # Setting for attention mechanism
        timestep = encoder_outputs.size(0) # (max_caption_length)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1) # (batch_size, max_caption_length, hidden)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (batch_size, max_caption_length, hidden)
        # Calculate attention energy score
        attn_energies = self.score(h, encoder_outputs) # (max_caption_length, batch_size)
        return torch.softmax(attn_energies, dim=1).unsqueeze(1) # (batch_size, 1, max_caption_length)

    def score(self, hidden, encoder_outputs):
        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # (batch_size, max_caption_length, hidden)
        energy = (energy @ self.v).squeeze(2) # (batch_size, max_caption_length)
        # Old codes
        #energy = energy.transpose(1, 2)  # [B*H*T]
        #v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        #energy = torch.bmm(v, energy)  # [B*1*T]
        return energy

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, pad_idx=0, dropout=0.0, embedding_dropout=0.0):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.dropout = dropout
        self.embedding_dropout = embedding_dropout

        self.embed = nn.Embedding(output_size, embed_size, padding_idx=pad_idx)
        self.attention = Attention(hidden_size)
        self.sru = nn.GRU(hidden_size + embed_size, hidden_size, num_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = F.dropout(self.embed(input), p=self.embedding_dropout, inplace=True).unsqueeze(0)  # (1, batch_size, embed_size)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs) # (max_caption_length, 1, batch_size)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))  # (batch_size, 1, embed_size)
        context = context.transpose(0, 1) # (1, batch_size, embed_size)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2) # (1, batch_size, embed + hidden)
        output, hidden = self.sru(rnn_input, last_hidden) # (1, batch_size, hidden)
        # context = context.squeeze(0)
        # output = self.out(torch.cat([output, context], 1))
        output = self.out(output.squeeze(0))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights