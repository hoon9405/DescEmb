import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.layers import OutPutLayer
"""
To-Do : (max, avg) pooling 어떻게 할지? 
        n_layers, dropout 도 argument로 받을 것인지
"""


class RNNModels(nn.Module):
  def __init__(self, args):
    super(RNNModels, self).__init__()
    self.visualize = args.visualize
    self.rnn_type = args.rnn_type
    self.pred_hidden_dim = args.pred_hidden_dim
    self.RNN_SESTS = args.RNN_SESTS
    self.rnn_att = args.rnn_att
    self.n_layers = args.rnn_layer

    pred_embed_dim = args.pred_embed_dim
    num_directions, self.bidirectional = (2, True) if self.rnn_type in ['biGRU', 'biLSTM'] else (1, False)
    dropout = args.dropout
  
    if args.rnn_type in ['uniGRU','biGRU']:
        self.model = nn.GRU(pred_embed_dim, self.pred_hidden_dim, dropout=dropout, batch_first=True, bidirectional=self.bidirectional, num_layers=self.n_layers)
    elif args.rnn_type in ['uniLSTM','biLSTM']:
        self.model = nn.LSTM(pred_embed_dim, self.pred_hidden_dim, dropout=dropout, batch_first=True, bidirectional=self.bidirectional, num_layers=self.n_layers)


    self.attn = nn.Linear(self.pred_hidden_dim, 1)
    self.attn_soft = nn.Softmax(dim=1)

    self.n_heads = 8
    self.multi_attn = nn.Linear(self.pred_hidden_dim, self.n_heads)
    self.multi_attn_soft = nn.Softmax(dim=2)
    self.combine_heads = nn.Linear(self.n_heads * self.pred_hidden_dim, self.pred_hidden_dim)

    self.output_fc = OutPutLayer(args.target, self.pred_hidden_dim*num_directions)


  def forward(self, x, lengths):
    self.model.flatten_parameters()
    self.B = x.size(0)

    if self.RNN_SESTS:
        x = x.sum(dim=2)  # (batch size x seq max length x max codes/step x embedding dim) --> (batch size x seq max length x embedding dim)

    output_seq, output_len=self.pack_pad_seq(x, lengths)

    if self.rnn_att == 'single_att':
      x = self.rnn_attention(output_seq, output_len)
    elif self.rnn_att == 'multi_att':
      x = self.rnn_multi_attention(output_seq, output_len)
    elif self.rnn_att == 'no_att':
      i = range(x.size(0)) 
      x = self.rnn_hidden_output(output_seq, i, output_len)
    if self.visualize == False:
        output = self.output_fc(x)
    else:
        output = x
    return output


  def pack_pad_seq(self, x, lengths):
      lengths = lengths.squeeze(-1).cpu()
    
        # explicit error: RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor
      #total_length = x.size(1) # max seq len -- see docs in https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
      packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
      output, _ = self.model(packed)
      output_seq, output_len = pad_packed_sequence(output, batch_first=True, padding_value=0)
      return output_seq, output_len
#total_length=total_length

  def rnn_attention(self, output_seq, output_len):
    attn_weights = self.attn(output_seq)  # output_seq = (B x S x H) ; attn_weights = (B x S x 1)

    ## add -inf after padding begins --> then after softmax, attention is 0
    max_len = attn_weights.size(1)
    mask = torch.arange(max_len)[None, :] < output_len[:, None]
    attn_weights[~mask] = float('-inf')

    attn_weights = self.attn_soft(attn_weights)  # attn_weights = (B x S x 1)

    attended = torch.mul(output_seq, attn_weights)  # attended = (B x S x H)
    summed = attended.sum(dim=1)  # summed = (B x H)
                     
    return summed


  def rnn_multi_attention(self, output_seq, output_len):
    attn_weights = self.multi_attn(output_seq)  # output_seq = (B x S x H) ; attn_weights = (B x S x n_heads)
    attn_weights = attn_weights.transpose(1, 2).view(output_seq.size(0), self.n_heads, output_seq.size(1),1)  # explicitly defined as a sanity check -- (B x n_heads x S x 1)

    ## add -inf after padding begins --> then after softmax, attention is 0
    max_len = attn_weights.size(2)  # S
    mask = torch.arange(max_len)[None, :] < output_len[:, None]  # (B x S)
    mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1)  # (B x S) --> (B x 1 x S) --> (B x n_heads x S)
    attn_weights[~mask] = float('-inf')

    attn_weights = self.multi_attn_soft(attn_weights)  # attn_weights = (B x n_heads x S x 1) **check here **

    repeat_output_seq = output_seq.unsqueeze(1).repeat(1, self.n_heads,1,1)  # (B x S x H) --> (B x 1 x S x H) --> (B x n_heads x S x H)
    attended = torch.mul(repeat_output_seq, attn_weights)  # attended = (B x n_heads x S x H)
    summed = attended.sum(dim=2)  # summed = (B x n_heads x H)

    concat = summed.view(self.B, self.n_heads * self.pred_hidden_dim)  # (B x n_heads x H) -->  (B x n_heads*H)
    combined = self.combine_heads(concat)  # (B x n_heads*H) --> (B x H)

    return combined


  def rnn_hidden_output(self, output_seq, i, lengths):
      if self.bidirectional:
        forward_output = output_seq[i, lengths-1, :self.pred_hidden_dim]
        backward_output = output_seq[:, 0, self.pred_hidden_dim:]
        output_hidden = torch.cat((forward_output, backward_output), dim=-1)
      else:
        output_hidden = output_seq[i, lengths -1, :]
      return output_hidden