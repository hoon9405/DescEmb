import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import numpy as np
from models.layers import SubWord_InPutLayer
from utils.trainer_utils import *

class TextEncoder_RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
     
        self.text_embed_dim = args.text_embed_dim
        self.text_hidden_size = args.text_hidden_size
        self.pred_embed_dim = args.pred_embed_dim
        self.subword_embed_layer = SubWord_InPutLayer(args)
        self.mlm_prob = args.textencoder_mlm_probability
        self.concat_type = args.concat_type
        self.model = nn.GRU(self.text_embed_dim, self.text_hidden_size, num_layers=1, dropout=args.dropout, batch_first=True, bidirectional=True)
        
        if self.concat_type =='concat_c':
            self.type_embedding =nn.Embedding(28, self.text_embed_dim)
        if args.textencoder_mlm_probability > 0.:
            self.mlm_linear = nn.Linear(self.text_hidden_size * 2, 28996)
        self.compress_fc = nn.Linear(2 *self.text_hidden_size, self.pred_embed_dim)


    def forward(self, x):
        if self.mlm_prob > 0.:
            return self.mlm_forward(x)
        else:
            return self.prediction_forward(x)

    def prediction_forward(self, x):
        self.model.flatten_parameters()
        if self.concat_type =='concat_c':
            type_ids = x['token_type_ids']
            x= x['input_ids'] 
            bsz, _, word_max_len = x.shape
            type_ids = type_ids.reshape(-1, word_max_len)
            type_ids = self.type_embedding(type_ids)

        bsz, _, word_max_len = x.shape

        x = x.reshape(-1, word_max_len)    # (B, S, W) -> (BXS, W)
        lengths = torch.argmin(x, dim=1)
        lengths = torch.where(lengths >0 , lengths,  1).detach().cpu()  # Pad length 0 -> 1
        x = self.subword_embed_layer(x)
        if self.concat_type =='concat_c':
          x = x + type_ids
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.model(packed)
        output_seq, _ = pad_packed_sequence(output, batch_first=True)

        i = range(x.size(0))
        forward_output = output_seq[i, lengths-1, :self.text_hidden_size]
        backward_output = output_seq[:, 0, self.text_hidden_size:]
        output = torch.cat((forward_output, backward_output), dim=-1)

        output = self.compress_fc(output)
        output = output.reshape(bsz, -1, self.pred_embed_dim)

        return output

    def mlm_forward(self, x):
        #bsz, _, word_max_len = x.shape
        
        bsz, word_max_len = x.shape

        
        x = x.reshape(-1, word_max_len)  # (B, S, W) -> (BXS, W)
        lengths = torch.argmin(x, dim=1)
        lengths = torch.where(lengths > 0, lengths, x.size(-1)).detach().cpu()  # Pad length 0 -> 1

        x = self.subword_embed_layer(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.model(packed)
        output_seq, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.)

        mlm_output = self.mlm_linear(output_seq)    # BX150, max_len, 288...
        diff_seq = word_max_len - mlm_output.size(1)
        padding = mlm_output.new_zeros(mlm_output.size(0), diff_seq, mlm_output.size(2))
        mlm_output = torch.cat((mlm_output, padding), dim=1)

        i = range(x.size(0))
        forward_output = output_seq[i, lengths - 1, :self.text_hidden_size]
        backward_output = output_seq[:, 0, self.text_hidden_size:]
        output = torch.cat((forward_output, backward_output), dim=-1)

        output = self.compress_fc(output)
        output = output.reshape(bsz, -1, self.pred_embed_dim)
        return output, mlm_output


