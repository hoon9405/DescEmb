import torch
import torch.nn as nn
import contextlib
import os
import pickle

class CodeInputLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        index_size_dict = {
            'nonconcat' : {
                'mimic' : 1889, 
                'eicu' : 1534, 
                'both' : 3223
            },
            'concat_a' : {
                'mimic' : 70873, 
                'eicu' : 34424, 
                'both' : 104353
            },
            'concat_b' : {
                'mimic' : 70873, 
                'eicu' : 34424,
                'both' : 104353
            },
            'concat_c' : {
                'mimic' : 70873, 
                'eicu' : 34424, 
                'both' : 104353
            },
            'concat_d' : {
                'mimic' : 3850,
                'eicu' : 4354,
                'both' : 8095
            }
        } 

        index_size = index_size_dict[args.concat_type][args.source_file]

        self.embedding =nn.Embedding(index_size, args.pred_embed_dim, padding_idx=0)

    def forward(self, x):
        output = self.embedding(x)
        return output

class SubwordInputLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        text_embed_dim = args.text_embed_dim

        # index_size = subword_index_size_dict[concat_type][source_file]
        index_size = 28996
        self.embedding =nn.Embedding(index_size, text_embed_dim, padding_idx=0)

    def forward(self, x):
        output = self.embedding(x)

        return output

class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, source):
        return source