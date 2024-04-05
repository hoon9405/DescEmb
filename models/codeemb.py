import logging

import torch
import torch.nn as nn
import os
import pickle
from models import register_model

logger = logging.getLogger(__name__)

@register_model("codeemb")
class CodeEmb(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        if args.src_data == 'pooled':
            mimic_dict = self.vocab_load(args.input_path, 'mimiciii', args.value_mode)
            eicu_dict = self.vocab_load(args.input_path, 'eicu', args.value_mode)
            index_size = len(mimic_dict) + len(eicu_dict) - 3
        else:
            vocab_dict = self.vocab_load(args.input_path, args.src_data, args.value_mode)
            index_size = len(vocab_dict)
        

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

    def vocab_load(self, data_path, src_data, value_mode):
        vocab_path = os.path.join(
            data_path, src_data, f'code_index_{value_mode}_vocab.pkl'
            )
        with open(vocab_path, 'rb') as file:
            vocab_dict = pickle.load(file)
        return vocab_dict
    
    def get_logits(self, net_output):
        return net_output.float()
    
    def get_targets(self, sample):
        return sample['label'].float()

    def forward(self, input_ids, **kwargs):
        output = self.embedding(input_ids)
        return output