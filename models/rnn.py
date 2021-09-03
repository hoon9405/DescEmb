import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.layers import OutputLayer

"""
To-Do : (max, avg) pooling 어떻게 할지? 
        n_layers, dropout 도 argument로 받을 것인지
"""

class RNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.visualize = args.visualize
        self.rnn_type = args.rnn_type
        self.pred_hidden_dim = args.pred_hidden_dim
        self.RNN_SESTS = args.RNN_SESTS
        self.rnn_att = args.rnn_att
        self.n_layers = args.rnn_layer

        pred_embed_dim = args.pred_embed_dim
        dropout = args.dropout

        self.model = nn.GRU(
            pred_embed_dim,
            self.pred_hidden_dim,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
            num_layers=self.n_layers
        )

        self.final_proj = nn.Linear(
            self.pred_hidden_dim,
            18 if args.target == 'diagnosis' else 1
        )


    def forward(self, x, lengths):
        self.model.flatten_parameters()

        output_seq, output_len=self.pack_pad_seq(x, lengths)

        i = range(x.size(0)) 
        x = output_seq[i, lengths - 1, :]
        x = self.rnn_hidden_output(output_seq, i, output_len)

        output = self.final_proj(x)

        return output

    def pack_pad_seq(self, x, lengths):
        lengths = lengths.squeeze(-1).cpu()
        
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.model(packed)
        output_seq, output_len = pad_packed_sequence(output, batch_first=True, padding_value=0)
        return output_seq, output_len