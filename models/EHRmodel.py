import torch
import torch.nn as nn
import torch.nn.functional as F

import os


class EHRModel(nn.Module):
  def __init__(self, args):
    super(EHRModel, self).__init__()

    self.args = args

    self.val_proj = None
    self.final_proj = None
    
    if args.concat_type == 'CV':
      self.val_proj = nn.Linear(1, args.pred_embed_dim)
      self.compress_fc = nn.Linear(args.pred_embed_dim*2, args.pred_embed_dim)
    self.embed_model = args.embed_model(args)

    if args.finetune and args.embed_model_mode == 'Scratch-RNN' and not args.transfer_fewshot:
      ckpt = torch.load(os.path.join(args.path, 'Scratch-RNN', f'mlm_predictive_rnn_{args.source_file}_rnn_unique_0.3_400.pt'))
      self.embed_model.load_state_dict(ckpt['rnntextencoder_state_dict'], strict=False)
      print('\n Load parameter from', os.path.join(args.path, 'Scratch-RNN', f'predictive_rnn_{args.source_file}_rnn_unique_0.3_400.pt'))
      print('\n Loaded RNN MLM embedding parameter')

    self.pred_model = args.pred_model(args)

  def forward(self, x, var, value):   
    '''
    RNN -> var = lengths 
    Transformer -> var = offset_order
    '''
    x = self.embed_model(x)  # (B, S, E)
    
    if self.val_proj is not None:
      # value shape is expected to be (B, S, 1)
      value = value.unsqueeze(dim=2)
      val_emb = self.val_proj(value)
      
      x = torch.cat((x, val_emb), dim=2)
      x = self.compress_fc(x)
    output = self.pred_model(x, var)
    return output


