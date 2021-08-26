import torch
import torch.nn as nn
import contextlib
import os
import pickle

class Code_Inputlayer(nn.Module):
    def __init__(self, args):
        super(Code_Inputlayer, self).__init__()
        source_file = args.source_file
        pred_embed = args.pred_embed_dim
        concat_type = args.concat_type
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
                          # concat version
        
        index_size = index_size_dict[concat_type][source_file]

        self.embedding =nn.Embedding(index_size, pred_embed, padding_idx=0)

    def forward(self, x):
        output = self.embedding(x)
        return output

class Desc_InputLayer(nn.Module):
    def __init__(self, args):
        super(Desc_InputLayer, self).__init__()
        self.cls_freeze = True if args.embed_model_mode =='DescEmb-FR' else False
        source_file = args.source_file
        bert_model = args.bert_model
        pred_embed_dim = args.pred_embed_dim     
        
        vocab_dirname = 'embed_vocab_file'
        if args.concat_type !='nonconcat':
          vocab_dirname += '_' + args.concat_type
        if args.predictive_SSL is not None:
          vocab_dirname +='_ssl'
        if args.bert_hidden_from_cls is True:
          vocab_dirname +='_hidden'
        
        embed_path = os.path.join(args.input_path, vocab_dirname, '{}_all_12_{}_cls_initialized.pkl'
                                  .format(source_file, bert_model))
        print('DescEmb load vocab_from : ', embed_path)
        initial_embed_weight = torch.load(open(embed_path, 'rb'))
        
        #[MASK], [CLS]  re initialize with torch random seed
        mask_embed = torch.randn(1, initial_embed_weight.size(1))
        cls_embed = torch.randn(1, initial_embed_weight.size(1))
        initial_embed_weight[1], initial_embed_weight[2] = mask_embed, cls_embed

        self.bert_embed =nn.Embedding(initial_embed_weight.size(0), initial_embed_weight.size(1), 
                                      _weight=initial_embed_weight)

        self.compress_fc = nn.Linear(initial_embed_weight.size(1), args.pred_embed_dim)

    def forward(self, x):
      with torch.no_grad() if self.cls_freeze else contextlib.ExitStack():
        x = self.bert_embed(x)

      output = self.compress_fc(x)

      return output


class OutPutLayer(nn.Module):
    def __init__(self, target, hidden_dim):
        super(OutPutLayer, self).__init__()

        output_size = 18 if target=='diagnosis' else 1

        self.output_layer=nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        output = self.output_layer(x)
        return output


class SubWord_InPutLayer(nn.Module):
    def __init__(self, args):
        super(SubWord_InPutLayer, self).__init__()
        source_file = args.source_file
        text_embed_dim = args.text_embed_dim
        concat_type = args.concat_type
        #subword should check  (we don't use)

        subword_index_size_dict = {
                      'nonconcat' : {
                    'mimic' : 1383, 
                    'eicu' : 895, 
                    'both' : 1559
                        },                  
                      'concat_a' : {
                    'mimic' : 1666, 
                    'eicu' : 900, 
                    'both' : 1803
                        },
                      'concat_b' : {
                    'mimic' : 1459, 
                    'eicu' : 861, 
                    'both' : 1593
                        },
                    'concat_c' : {
                  'mimic' : 1459, 
                  'eicu' : 861, 
                  'both' : 1593
                      },
                      }
                          # concat version


        # index_size = subword_index_size_dict[concat_type][source_file]
        index_size = 28996
        self.embedding =nn.Embedding(index_size, text_embed_dim, padding_idx=0)

    def forward(self, x):
        output = self.embedding(x)

        return output
