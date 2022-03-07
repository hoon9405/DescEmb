import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models import register_model

@register_model("rnn")
class RNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_embed_dim = args.pred_embed_dim
        self.pred_hidden_dim = args.pred_hidden_dim
        self.n_layers = args.rnn_layer

        self.model = nn.GRU(
            self.pred_embed_dim,
            self.pred_hidden_dim,
            dropout=args.dropout,
            batch_first=True,
            bidirectional=False,
            num_layers=self.n_layers
        )

        self.final_proj = nn.Linear(
            self.pred_hidden_dim,
            18 if args.task == 'diagnosis' else 1
        )

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, x, seq_len, **kwargs):
        # TODO: used for DataParallel (not sure what it is..)
        self.model.flatten_parameters()

        output_seq, _ = self.pack_pad_seq(x, seq_len)

        i = range(x.size(0))
        x = output_seq[i, -1, :]

        breakpoint()
        
        output = self.final_proj(x)

        breakpoint()

        return output

    def pack_pad_seq(self, x, lengths):
        lengths = lengths.squeeze(-1).cpu()
        # x.shape: torch.Size([128, 150, 128])
        # lengths.shape: torch.Size([128]) 
        breakpoint()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # packed[0].shape (data): [8043, 128]
        # packed[1].shape (batch_sizes): [150]
        # packed[2].shape (sorted_indices): [128]
        # packed[3].shape (unsorted_indices): [128]
        breakpoint()
        output, _ = self.model(packed)
        # output[0].shape (data): [8043, 256]
        # output[1].shape (batch_sizes): [150]
        # output[2].shape (sorted_indices): [128]
        # output[3].shape (unsorted_indices): [128]
        breakpoint()
        output_seq, output_len = pad_packed_sequence(output, batch_first=True, padding_value=0)
        # output_seq.shape: [128, 150, 256]
        # output_len.shape: [128]
        breakpoint()
        return output_seq, output_len