import torch.nn as nn

class SubwordInputLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        enc_embed_dim = args.enc_embed_dim

        # index_size = subword_index_size_dict[concat_type][source_file]
        index_size = 28996
        self.embedding =nn.Embedding(index_size, enc_embed_dim, padding_idx=0)

    def forward(self, x):
        output = self.embedding(x)

        return output