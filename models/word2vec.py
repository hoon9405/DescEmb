import torch
import torch.nn as nn

from models import register_model

# reference: https://github.com/blackredscarf/pytorch-SkipGram/blob/a9fa5a888a7b0c6170eb1fe146e59f54041b2613/model.py

@register_model(name="word2vec")
class Word2VecModel(nn.Module):
    """
    Word2Vec in skipgram
    """
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)

        initrange = (2.0 / (vocab_size + emb_dim)) ** 0.5 # Xavier init
        self.input_emb.weight.data.uniform_(-initrange, initrange)
        self.output_emb.weight.data.uniform_(-0, 0)

    def forward(self, target_input, context, neg):
        # target_input (B)  context (B)  neg (B, neg_size)
        target_input = target_input.squeeze(-1)
        v = self.input_emb(target_input)
        u = self.output_emb(context)
        pos_val = torch.bmm(u, v.unsqueeze(2)).squeeze(2)
        positive_val = nn.LogSigmoid()(torch.sum(pos_val, dim=1)).squeeze()

        # u_hat : (B, neg_size, emb_dim)
        u_hat = self.output_emb(neg)
        neg_vals = torch.bmm(u_hat, v.unsqueeze(2)).squeeze(2)
        neg_val = nn.LogSigmoid()(-torch.sum(neg_vals, dim=1)).squeeze()

        loss = positive_val + neg_val
        return -loss.mean()