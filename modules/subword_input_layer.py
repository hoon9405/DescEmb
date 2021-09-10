import torch.nn as nn

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