import os
import logging

import torch
from torch._C import Value
import torch.nn as nn

from models import register_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)

@register_model("ehr_model")
class EHRModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.transfer and args.load_pretrained_weights:
            logger.warn(
                "--transfer and --load_pretrained_weights are set simultaneously. "
                "--load_pretrained_weights is ignored."
            )
            args.load_pretrained_weights = False

        self.val_proj = None
        self.final_proj = None
        
        if args.value_embed_type == 'VC':
            self.val_proj = nn.Linear(1, args.pred_embed_dim)
            self.post_embed_proj = nn.Linear(args.pred_embed_dim*2, args.pred_embed_dim)

        self.embed_model = self._embed_model.build_model(args)
        self.pred_model = self._pred_model.build_model(args)

        if args.transfer:
            logger.info(
                f"Preparing to transfer pre-trained model {args.model_path}"
            )
            loaded_checkpoint = torch.load(args.model_path)
            loaded_state_dict = loaded_checkpoint['model_state_dict']
            loaded_args = loaded_checkpoint['args']

            assert (
                loaded_args.embed_model == self.args.embed_model
                and loaded_args.pred_model == self.args.pred_model
            ), (
                'found mismatch with transferred model. please check if '
                'are trying to transfer model that has a different architecture.\n'
                f'transferred embed_model: {loaded_args.embed_model}\n'
                f'self.embed_model: {self.args.embed_model}\n'
                f'transferred pred_model: {loaded_args.pred_model}\n'
                f'self.pred_model: {self.args.pred_model}'
            )

            state_dict = {
                k: v for k,v in loaded_state_dict.items() if (
                    args.embed_model.startswith('codeemb')
                    and 'embedding' not in k
                )
            }
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if unexpected or len(missing) > 1:
                logger.warn(
                    'transferred model has unexpected or missing keys.'
                )
    @property
    def _embed_model(self):
        return MODEL_REGISTRY[self.args.embed_model]
    
    @property
    def _pred_model(self):
        return MODEL_REGISTRY[self.args.pred_model]

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def get_logits(self, net_output):
        return net_output.float()

    def get_targets(self, sample):
        return sample['label'].float()

    def forward(self, value, **kwargs):
        x = self.embed_model(**kwargs)  # (B, S, E)
        
        if self.val_proj:
            # value shape is expected to be (B, S, 1)
            value = value.unsqueeze(dim=2)
            val_emb = self.val_proj(value)
            
            x = torch.cat((x, val_emb), dim=2)
            x = self.post_embed_proj(x)

        net_output = self.pred_model(x, **kwargs)
        return net_output