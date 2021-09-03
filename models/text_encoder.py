import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import AutoConfig, AutoModel
from utils.trainer_utils import text_encoder_load_path, text_encoder_load_model

from models.layers import SubwordInputLayer, IdentityLayer

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

        if args.embed_model_mode == 'BERT-Scr':   #Loading Huggingface model with random initialized
            config = AutoConfig.from_pretrained(bert_model_config[args.bert_model][0])
            self.model = AutoModel.from_config(config)
        elif args.embed_model_mode == 'BERT-FT':  #Loading Huggingface model with pre-trained parameters
            self.model = AutoModel.from_pretrained(bert_model_config[args.bert_model][0])
        elif args.embed_model_mode == 'BERT-CLS-FT':
            with torch.no_grad():
                self.model = AutoModel.from_pretrained(bert_model_config[args.bert_model][0])
            self.model = nn.ModuleList(self.model, IdentityLayer())

        self.mlm_proj = None
        if args.embed_model_mode.startswith('MLM-pretrain'):
            self.mlm_proj = nn.Linear(bert_model_config[args.bert_model][1], 28996)
        self.post_encode_proj = nn.Linear(bert_model_config[args.bert_model][1], self.pred_embed_dim)

        if args.concat_type =='DSVA+DPE':
            old_token_type_embeddings = self.model.embeddings.token_type_embeddings
            new_token_type_embeddings = self.model._get_resized_embeddings(old_token_type_embeddings, 28)
            self.model.embeddings.token_type_embeddings = new_token_type_embeddings

        if args.load_pretrained_weights:
            assert args.model_path, args.model_path
            state_dict = torch.load(args.model_path)['model_state_dict']
            #XXX
            state_dict = {
                k: v.lstrip('bert.')
                for k, v in state_dict.items() if 'cls' not in k
            }
            self.model.load_state_dict(state_dict, strict=True)
            print("Model fully loaded!")

    def forward(self, net_input):
        bsz, _, word_max_len = net_input['input_ids'].shape

        net_input = {
            k: v.view(-1, word_max_len)
            for k, v in net_input.items()
        }

        bert_outputs = self.model(**net_input)

        net_output = (
            self.post_encode_proj(
                bert_outputs[0][:, 0, :]
            ).view(bsz, -1, self.pred_embed_dim)
        )

        mlm_output = None
        if self.mlm_proj:
            mlm_output = self.mlm_proj(bert_outputs[0]) # (B x S, W, H) -> (B x S, W, Bert-vocab)

        return net_output, mlm_output

class RNNTextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.text_embed_dim = args.text_embed_dim
        self.text_hidden_size = args.text_hidden_size
        self.pred_embed_dim = args.pred_embed_dim
        self.subword_embed_layer = SubwordInputLayer(args)
        self.concat_type = args.concat_type
        self.model = nn.GRU(
            self.text_embed_dim,
            self.text_hidden_size,
            num_layers=1,
            dropout=args.dropout,
            batch_first=True,
            bidirectional=True
        )

        self.value_embedding = None
        self.mlm_proj = None

        if args.embed_model_mode.startswith('MLM-pretrain'):
            self.mlm_proj = nn.Linear(self.text_hidden_size * 2, 28996)

        if self.mlm_proj is None and self.concat_type == 'DSVA+DPE':
            self.value_embedding = nn.Embedding(28, self.text_embed_dim)

        self.post_encode_proj = nn.Linear(self.text_hidden_size * 2, self.pred_embed_dim)
    
        if args.load_pretrained_weights:
            assert args.model_path, args.model_path
            state_dict = torch.load(args.model_path)['model_state_dict']
            self.model.load_state_dict(state_dict, strict=True)
            print("Model fully loaded!")

    def forward(self, net_input):
        if self.mlm_proj is None:
            self.model.flatten_parameters()
            if self.value_embedding:
                x = net_input['input_ids']
                type_ids = self.type_embedding(
                    net_input['token_type_ids'].view(-1, x.size(-1))
                )
            x = x.view(-1, x.size(-1))
        else:
            # XXX x.shape: (bsz, word_max_len)???
            x = net_input

        bsz, word_max_len = x.shape

        lengths = torch.argmin(x, dim=1)
        lengths = torch.where(lengths > 0, lengths, x.size(-1)).detach().cpu()

        x = self.subword_embed_layer(x)

        if self.value_embedding:
            x = x + type_ids
        
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.model(packed)
        output_seq, _ = pad_packed_sequence(output, batch_first=True, padding_value=0)

        mlm_output = None
        if self.mlm_proj:
            mlm_output = self.mlm_proj(output_seq)
            diff_seq = word_max_len - mlm_output.size(1)
            padding = mlm_output.new_zeros(mlm_output.size(0), diff_seq, mlm_output.size(2))
            mlm_output = torch.cat((mlm_output, padding), dim=1)
        
        i = range(bsz)
        forward_output = output_seq[i, lengths - 1, :self.text_hidden_size]
        backward_output = output_seq[:, 0, self.text_hidden_size:]
        net_output = torch.cat((forward_output, backward_output), dim=-1)

        net_output = (
            self.post_encode_proj(
                net_output
            ).view(bsz, -1, self.pred_embed_dim)
        )

        return net_output, mlm_output