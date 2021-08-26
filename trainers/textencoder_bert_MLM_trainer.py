import os
import wandb
import torch
import torch.nn as nn
from utils.trainer_utils import *
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoConfig, BertForMaskedLM
from dataset.dataset import *

class TextEncoderBert_MLM_Trainer():
    def __init__(self, args, device):
        self.word_max_length = args.word_max_length
        self.device = device
        self.debug = args.debug
        self.source_file = args.source_file
        self.mlm_prob = args.textencoder_mlm_probability
        self.pred_model_mode = args.pred_model_mode
        self.embed_model_mode = args.embed_model_mode
        self.epoch_list = [5, 10, 25, 50, 100, 200, 400, 600, 800, 1000]

        if not self.debug:
            wandb.init(project=args.wandb_project_name, entity='pretrained_ehr', config=args, reinit=True)

        lr = args.lr

        self.n_epochs = args.n_epochs
        filename = '{}_{}_{}_{}_{}_{}_{}'.format(args.source_file, args.bert_model, args.lr, args.batch_size, 
                                                    args.textencoder_ssl, args.textencoder_mlm_scratch, args.bert_hidden_from_cls)
        self.path = os.path.join(args.path, 'SSL', 'MLM', 'pretrain', filename)
        print('Model will be saved in {}'.format(self.path))
        
        self.dataset = TokenizedDataset
        self.train_dataloader = DataLoader(self.dataset(args, 'train'), batch_size = args.batch_size, num_workers=8, shuffle = True)
    
        self.early_stopping = EarlyStopping(patience=20, verbose=True)
        bert_model_config = {'bert': ["bert-base-uncased", 768],
                             'bio_clinical_bert': ["emilyalsentzer/Bio_ClinicalBERT", 768],
                             'pubmed_bert': ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 768],
                             'blue_bert': ["bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", 768],
                             'bio_bert': ["dmis-lab/biobert-v1.1", 768],
                             'bert_tiny': ["google/bert_uncased_L-2_H-128_A-2", 128],
                             'bert_mini': ["google/bert_uncased_L-4_H-256_A-4", 256],
                             'bert_small': ["google/bert_uncased_L-4_H-512_A-8", 512]}

        # BertForMaksedLM 
        if args.textencoder_mlm_scratch is True:
          config = AutoConfig.from_pretrained(bert_model_config[args.bert_model][0])
          self.model = BertForMaskedLM(config).to(self.device)
        else:
          self.model = BertForMaskedLM.from_pretrained(bert_model_config[args.bert_model][0]).to(self.device)

        # optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        num_params = count_parameters(self.model)
        print('Number of parameters: {}'.format(num_params))

    def parse_sample(self, sample):
        if self.embed_model_mode in ['Scratch-Bert', 'Finetune-Bert','pretrain']:
          x={}
          x['input_ids'] = sample['input'].to(self.device)
          x['token_type_ids'] = sample['token_type_ids'].to(self.device)
          x['attention_mask'] = sample['attention_mask'].to(self.device)
        
        else:
          x = sample['input'].to(self.device)
        
        return x
    
    def train(self):
        best_avg_mlm_loss = 0.0
        best_avg_mlm_acc = 0.0

        for n_epoch in range(self.n_epochs + 1):

            avg_mlm_acc = 0.0
            avg_mlm_loss = 0.0
            self.model.train()
            print('epoch :', n_epoch)
            for iter, sample in tqdm.tqdm(enumerate(self.train_dataloader)):
                self.optimizer.zero_grad(set_to_none=True)            
                x = self.parse_sample(sample)
                x['labels']= sample['mlm_labels'].to(self.device) 
                output = self.model(**x)
          
                # input shape = (B,SXW)
                mlm_loss_iter = output.loss
                logits = output.logits

                avg_mlm_loss += mlm_loss_iter.item() / len(self.train_dataloader)

                # compute accuracy
                total_mlm_correct = 0
                total_mlm_element = 0

                eq = (logits.argmax(dim=-1).eq(x['labels'])).cpu().numpy()
                label_np = x['labels'].cpu().numpy()

                for bs, label_i in enumerate(label_np):
                    index = np.where(label_i == -100)[0]
                    f_label = np.delete(label_i, index)
                    f_eq = np.delete(eq[bs], index)
                    total_mlm_correct += f_eq.sum()
                    total_mlm_element += len(f_label)

                if total_mlm_element == 0 or type(total_mlm_element) != int:
                    print('total_mlm_element',total_mlm_element)
                    print('index', index)
                    print('f_label', f_label)

                acc_iter = total_mlm_correct / total_mlm_element
                avg_mlm_acc += acc_iter / len(self.train_dataloader)

                mlm_loss_iter.backward()
                self.optimizer.step()

            if n_epoch in self.epoch_list:
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': avg_mlm_loss,
                                'acc': avg_mlm_acc,
                                'epochs': n_epoch}, self.path+'_{}.pt'.format(n_epoch))
                    print('Model parameter saved at epoch {} for transfer'.format(n_epoch))

            if not self.debug:
                wandb.log({'epoch': n_epoch,
                           'train_mlm_loss:': avg_mlm_loss,
                           'train_mlm_acc': avg_mlm_acc
                           })

            print('[Train]  loss: {:.3f},  acc: {:.3f}'.format(avg_mlm_loss, avg_mlm_acc))

            '''
            self.early_stopping(avg_eval_mlm_acc)
            if self.early_stopping.early_stop:
               print('Early stopping')

                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': best_avg_mlm_loss,
                                'acc': best_avg_mlm_acc,
                                'epochs': n_epoch}, self.final_path)

                break
            '''
