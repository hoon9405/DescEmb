import torch
import torch.nn as nn
import torch.distributions as dist
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#import pdb
import transformers
from transformers import AutoConfig, AutoModel
from utils.trainer_utils import count_parameters, text_encoder_load_path, text_encoder_load_model
import numpy as np
#from transformer_text import *

import math


class TextEncoder_BERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_embed_dim = args.pred_embed_dim
        self.max_len = args.max_length
        self.bert_hidden_from_cls = args.bert_hidden_from_cls
        self.mlm_prob = args.textencoder_mlm_probability
        
        bert_model_config = {'bert': ["bert-base-uncased", 768],
                             'bio_clinical_bert': ["emilyalsentzer/Bio_ClinicalBERT", 768],
                             'pubmed_bert': ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 768],
                             'blue_bert': ["bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", 768],
                             'bio_bert': ["dmis-lab/biobert-v1.1", 768],
                             'bert_tiny': ["google/bert_uncased_L-2_H-128_A-2", 128],
                             'bert_mini': ["google/bert_uncased_L-4_H-256_A-4", 256],
                             'bert_small': ["google/bert_uncased_L-4_H-512_A-8", 512]}
        
        if args.embed_model_mode == 'Scratch-Bert':   #Loading Huggingface model with random initialized
            config = AutoConfig.from_pretrained(bert_model_config[args.bert_model][0])
            self.model = AutoModel.from_config(config)
        elif args.embed_model_mode == 'Finetune-Bert':  #Loading Huggingface model with pre-trained parameters
            self.model = AutoModel.from_pretrained(bert_model_config[args.bert_model][0])

        if args.textencoder_mlm_probability > 0.:
            self.mlm_linear = nn.Linear(bert_model_config[args.bert_model][1], 28996)
        self.compress_fc = nn.Linear(bert_model_config[args.bert_model][1], self.pred_embed_dim) 

        if args.concat_type =='concat_c':
            old_token_type_embeddings = self.model.embeddings.token_type_embeddings
            new_token_type_embeddings = self.model._get_resized_embeddings(old_token_type_embeddings, 28)
            self.model.embeddings.token_type_embeddings = new_token_type_embeddings

        if args.transfer:
           model_path = text_encoder_load_path(args)
           self.model = text_encoder_load_model(model_path, self.model)

       

    def forward(self, x):
        if self.mlm_prob > 0.:
            return self.mlm_forward(x)
        else:
            return self.prediction_forward(x)

    def prediction_forward(self, x):
        bsz, _, word_max_len = x['input_ids'].shape

        # reshape (B, S, W) -> (B*S, W)
        x['input_ids'] = x['input_ids'].reshape(-1, word_max_len)
        x['token_type_ids'] = x['token_type_ids'].reshape(-1, word_max_len)
        x['attention_mask'] =x['attention_mask'].reshape(-1, word_max_len)

        bert_outputs= self.model(**x)   # cls_output shape (B * S, 128)
        # [0] all hidden_layers  [1] CLS hidden with tanh

        x = bert_outputs[0][:,0,:]   # (B X S, E) -> CLS from last hidden output  

        x = self.compress_fc(x)

        output = x.reshape(bsz, -1, self.pred_embed_dim)  # x, shape of (B, S, E)
        #print('pre-BERT forward calculation DOEN!')
        #print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')

        return output

    def mlm_forward(self, x):
        bsz, _, word_max_len = x['input_ids'].shape

        # (B, S, W) -> (BXS, W)
        x['input_ids'] = x['input_ids'].reshape(-1, word_max_len)
        x['token_type_ids'] = x['token_type_ids'].reshape(-1, word_max_len)
        x['attention_mask'] =x['attention_mask'].reshape(-1, word_max_len)

        bert_outputs= self.model(**x)   # cls_output shape (B * S, H)
        # [0] all hidden_layers  [1] CLS hidden with tanh
        x = bert_outputs[0][:,0,:]   # (B X S, 1, H) -> CLS from last hidden output
      
        x = self.compress_fc(x)
        output = x.reshape(bsz, -1, self.pred_embed_dim) #(B, S, H)

        mlm_output = self.mlm_linear(bert_outputs[0])    # (BXS, W, H) -> (BXS, W, Bert-vocab)

        return output, mlm_output


