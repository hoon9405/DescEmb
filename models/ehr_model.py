import torch
import torch.nn as nn

from models import register_model, MODEL_REGISTRY

@register_model("ehr_model")
class EHRModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.val_proj = None
        self.final_proj = None
        
        if args.concat_type == 'VC':
            self.val_proj = nn.Linear(1, args.pred_embed_dim)
            self.post_embed_proj = nn.Linear(args.pred_embed_dim*2, args.pred_embed_dim)

        self.embed_model = self._embed_model.build_model(args)
        self.pred_model = self._pred_model.build_model(args)
    
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