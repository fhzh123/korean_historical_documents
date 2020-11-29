# Import modules
import random

# Import PyTorch
import torch
import torch.nn.functional as F
from torch import nn

# Import custom modules
from encoder import Encoder
from decoder import Decoder, Attention

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, dropout=0.0):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(self.encoder.hidden_size, self.decoder.hidden_size)
        self.dropout = dropout
        self.device = device

    def forward(self, src, trg, king_id, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)
        # Encoding source sentences
        encoder_output, hidden = self.encoder(src, king_id)
        hidden = hidden[-self.decoder.n_layers:]
        # hidden = torch.tanh(self.encoder.linear(hidden))
        hidden = torch.tanh(self.linear(hidden))
        # Decoding
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            # Teacher forcing mechanism
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if is_teacher:
                output = Variable(trg.data[t].to(self.device))
            else:
                output = top1.to(self.device)
        return outputs