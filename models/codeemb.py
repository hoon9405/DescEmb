import logging

import torch
import torch.nn as nn

from models import register_model

logger = logging.getLogger(__name__)

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

        if not args.transfer and args.load_pretrained_weights:
            assert args.model_path, args.model_path
            logger.info(
                "Preparing to load pre-trained checkpoint {}".format(args.model_path)
            )
            state_dict = torch.load(args.model_path)['model_state_dict']

            state_dict = {
                'embedding.weight': v for k, v in state_dict.items()
                if k == 'input_emb.weight'
            }
            self.load_state_dict(state_dict, strict=True)

            logger.info(
                "Loaded checkpoint {}".format(
                    args.model_path
                )
            )

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