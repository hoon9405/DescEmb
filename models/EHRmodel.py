import torch
import torch.nn as nn

from . import RNNModel

class EHRModel(nn.Module):
  def __init__(self, args, embed_model):
    super().__init__()

    self.args = args

    self.val_proj = None
    self.final_proj = None
    
    if args.concat_type == 'VC':
      self.val_proj = nn.Linear(1, args.pred_embed_dim)
      self.post_embed_proj = nn.Linear(args.pred_embed_dim*2, args.pred_embed_dim)
    self.embed_model = embed_model(args)

    self.pred_model = RNNModel(args)

  def forward(self, x, var, value):   
    '''
    RNN -> var = lengths 
    Transformer -> var = offset_order
    '''
    x = self.embed_model(x)  # (B, S, E)
    
    if self.val_proj:
      # value shape is expected to be (B, S, 1)
      value = value.unsqueeze(dim=2)
      val_emb = self.val_proj(value)
      
      x = torch.cat((x, val_emb), dim=2)
      x = self.post_embed_proj(x)

    output = self.pred_model(x, var)
    return output