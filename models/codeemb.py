import torch.nn as nn

from models import register_model

@register_model("codeemb")
class CodeEmb(nn.Module):
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