import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import AutoConfig, AutoModel

from models import register_model
from modules import (
    SubwordInputLayer,
    IdentityLayer
)

logger = logging.getLogger(__name__)

@register_model("descemb_bert")
class BertTextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_embed_dim = args.pred_embed_dim

        bert_model_config = {'bert': ["bert-base-uncased", 768],
                             'bio_clinical_bert': ["emilyalsentzer/Bio_ClinicalBERT", 768],
                             'pubmed_bert': ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 768],
                             'blue_bert': ["bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", 768],
                             'bio_bert': ["dmis-lab/biobert-v1.1", 768],
                             'bert_tiny': ["google/bert_uncased_L-2_H-128_A-2", 128],
                             'bert_mini': ["google/bert_uncased_L-4_H-256_A-4", 256],
                             'bert_small': ["google/bert_uncased_L-4_H-512_A-8", 512]}

        if not args.init_bert_params:   #Loading Huggingface model with random initialized
            config = AutoConfig.from_pretrained(bert_model_config[args.bert_model][0])
            self.model = AutoModel.from_config(config)
        elif args.init_bert_params_with_freeze:
            with torch.no_grad():
                self.model = AutoModel.from_pretrained(bert_model_config[args.bert_model][0])
        else:  #Loading Huggingface model with pre-trained parameters
            self.model = AutoModel.from_pretrained(bert_model_config[args.bert_model][0])
            self.model = nn.ModuleList(self.model, IdentityLayer())

        self.mlm_proj = None
        if args.task == "mlm":
            self.mlm_proj = nn.Linear(bert_model_config[args.bert_model][1], 28996)
        self.post_encode_proj = nn.Linear(bert_model_config[args.bert_model][1], self.pred_embed_dim)

        if args.value_embed_type =='DSVA_DPE':
            old_token_type_embeddings = self.model.embeddings.token_type_embeddings
            new_token_type_embeddings = self.model._get_resized_embeddings(old_token_type_embeddings, 28)
            self.model.embeddings.token_type_embeddings = new_token_type_embeddings

        if not args.transfer and args.load_pretrained_weights:
            assert args.model_path, args.model_path
            logger.info(
                "Preparing to load pre-trained checkpoint {}".format(args.model_path)
            )
            state_dict = torch.load(args.model_path)['model_state_dict']
            state_dict = {
                k: v for k, v in state_dict.items()
                if 'mlm' not in k
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
        if net_output.dim() > 2:
            net_output = net_output.view(-1, net_output.size(-1))
        return net_output.float()

    def get_targets(self, sample):
        target = sample['label']
        if target.dim() > 1:
            target = target.view(-1)
        return target.long()

    def forward(self, input_ids, token_type_ids, attention_mask, **kwargs):
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
        bsz, _, word_max_len = input_ids.shape

        bert_args = {
            "input_ids": input_ids.view(-1, word_max_len),
            "token_type_ids": token_type_ids.view(-1, word_max_len),
            "attention_mask": attention_mask.view(-1, word_max_len)
        }

        bert_outputs = self.model(**bert_args)

        net_output = (
            self.post_encode_proj(
                bert_outputs[0][:, 0, :]
            ).view(bsz, -1, self.pred_embed_dim)
        )

        if self.mlm_proj:
            mlm_output = self.mlm_proj(bert_outputs[0]) # (B x S, W, H) -> (B x S, W, Bert-vocab)
            return mlm_output

        return net_output

@register_model("descemb_rnn")
class RNNTextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.enc_embed_dim = args.enc_embed_dim
        self.enc_hidden_dim = args.enc_hidden_dim
        self.pred_embed_dim = args.pred_embed_dim
        self.subword_embed_layer = SubwordInputLayer(args)
        self.value_embed_type = args.value_embed_type
        self.model = nn.GRU(
            self.enc_embed_dim,
            self.enc_hidden_dim,
            num_layers=1,
            dropout=args.dropout,
            batch_first=True,
            bidirectional=True
        )

        self.task = args.task
        self.value_embedding = None
        self.mlm_proj = None

        if args.task == "mlm":
            self.mlm_proj = nn.Linear(self.enc_hidden_dim * 2, 28996)

        if self.mlm_proj is None and self.value_embed_type == 'DSVA_DPE':
            self.value_embedding = nn.Embedding(28, self.enc_embed_dim)

        self.post_encode_proj = nn.Linear(self.enc_hidden_dim * 2, self.pred_embed_dim)
    
        if not args.transfer and args.load_pretrained_weights:
            assert args.model_path, args.model_path
            logger.info(
                "Preparing to load pre-trained checkpoint {}".format(args.model_path)
            )
            state_dict = torch.load(args.model_path)['model_state_dict']
            state_dict = {
                k: v for k, v in state_dict.items()
                if 'mlm' not in k
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
        if net_output.dim() > 2:
            net_output = net_output.view(-1, net_output.size(-1))
        return net_output.float()

    def get_targets(self, sample):
        target = sample['label']
        if target.dim() > 1:
            target = target.view(-1)
        return target.long()

    def forward(self, input_ids, token_type_ids, **kwargs):
        if self.mlm_proj is None:
            bsz, _, word_max_len = input_ids.shape

            self.model.flatten_parameters()
            if self.value_embedding:
                type_ids = self.type_embedding(
                    token_type_ids.view(-1, word_max_len)
                )
            input_ids = input_ids.view(-1, word_max_len)
        else:
            bsz, word_max_len = input_ids.shape

        lengths = torch.argmin(input_ids, dim=1)
        lengths = torch.where(lengths > 0, lengths, 1).detach().cpu()

        x = self.subword_embed_layer(input_ids)

        if self.value_embedding:
            x = x + type_ids
        
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.model(packed)
        output_seq, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)

        if self.mlm_proj:
            mlm_output = self.mlm_proj(output_seq)
            diff_seq = word_max_len - mlm_output.size(1)
            padding = mlm_output.new_zeros(mlm_output.size(0), diff_seq, mlm_output.size(2))
            mlm_output = torch.cat((mlm_output, padding), dim=1)
            return mlm_output
        
        forward_output = output_seq[:, -1, :self.enc_hidden_dim]
        backward_output = output_seq[:, 0, self.enc_hidden_dim:]
        net_output = torch.cat((forward_output, backward_output), dim=-1)

        net_output = (
            self.post_encode_proj(
                net_output
            ).view(bsz, -1, self.pred_embed_dim)
        )

        return net_output