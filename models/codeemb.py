import torch.nn as nn

from models import register_model

#XXX 2. w2v load
@register_model("codeemb")
class CodeEmb(nn.Module):
    def __init__(self, args):
        super().__init__()
        index_size_dict = {
            'nonconcat' : {
                'mimic' : 1889, 
                'eicu' : 1534, 
                'pooled' : 3223
            },
            'VA' : {
                'mimic' : 70873, 
                'eicu' : 34424, 
                'pooled' : 104353
            },
            'DSVA' : {
                'mimic' : 70873, 
                'eicu' : 34424,
                'pooled' : 104353
            },
            'DSVA_DPE' : {
                'mimic' : 70873, 
                'eicu' : 34424, 
                'pooled' : 104353
            },
            'VC' : {
                'mimic' : 3850,
                'eicu' : 4354,
                'pooled' : 8095
            }
        } 

        index_size = index_size_dict[args.value_embed_type][args.data]

        self.embedding =nn.Embedding(index_size, args.pred_embed_dim, padding_idx=0)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def get_logits(self, net_output):
        return net_output.float()
    
    def get_targets(self, sample):
        return sample['label'].float()

    def forward(self, input_ids, **kwargs):
        output = self.embedding(input_ids)
        return output