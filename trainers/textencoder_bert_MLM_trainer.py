import os
import wandb
import torch
import torch.nn as nn
import numpy as np
from utils.trainer_utils import *
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoConfig, BertForMaskedLM
from datasets.dataset import *
from models.textencoder_rnn import TextEncoder_RNN
from models.textencoder_bert import TextEncoder_BERT

class TextEncoderBert_MLM_Trainer():
    def __init__(self, args, device):
        self.device = device
        self.source_file = args.source_file
        self.mlm_prob = args.mlm_probability
        self.embed_model_mode = args.embed_model_mode
        self.epoch_list = [5, 10, 25, 50, 100, 200, 400, 600, 800, 1000]

        wandb.init(project=args.wandb_project_name, entity='pretrained_ehr', config=args, reinit=True)

        self.n_epochs = args.n_epochs
        # TODO: check
        filename = '{}_{}_{}_{}_{}'.format(args.source_file, args.text_encoder_model, args.lr, args.batch_size, args.load_bert_scratch)

        self.path = os.path.join(args.output_path, 'MLM', filename)
        print('Model will be saved in {}'.format(self.path))
        
        self.dataset = TokenizedDataset
        self.train_dataloader = DataLoader(self.dataset(args, 'train'), batch_size = args.batch_size, num_workers=8, shuffle = True)

        # For BERTs
        if self.embed_model_mode == 'MLM-pretrain-BERT':
            self.model = TextEncoder_BERT(args).cuda()

            # # BertForMaksedLM
            # if args.load_bert_scratch is True:
            #   config = AutoConfig.from_pretrained(bert_model_config[args.text_encoder_model][0])
            #   self.model = BertForMaskedLM(config).to(self.device)
            # else:
            #   self.model = BertForMaskedLM.from_pretrained(bert_model_config[args.text_encoder_model][0]).to(self.device)

        elif self.embed_model_mode == 'MLM-pretrain-RNN':
            self.model = TextEncoder_RNN(args).to(device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        num_params = count_parameters(self.model)
        print('Number of parameters: {}'.format(num_params))

    def parse_sample(self, sample):
        # TODO: BERT-Scr, BERT-FT는 왜있는 거지?
        if self.embed_model_mode in ['BERT-Scr', 'BERT-FT', 'MLM-pretrain-BERT', 'MLM-pretrain-RNN']:
          x={}
          x['input_ids'] = sample['input'].to(self.device)
          x['token_type_ids'] = sample['token_type_ids'].to(self.device)
          x['attention_mask'] = sample['attention_mask'].to(self.device)
        
        else:
          x = sample['input'].to(self.device)
        
        return x

    def train(self):
        if self.embed_model_mode == 'MLM-pretrain-BERT':
            self.bert_train()
        elif self.embed_model_mode == 'MLM-pretrain-RNN':
            self.rnn_train()
    
    def bert_train(self):
        for n_epoch in range(self.n_epochs + 1):

            avg_mlm_acc = 0.0
            avg_mlm_loss = 0.0
            self.model.train()
            print('epoch :', n_epoch)
            for iter, sample in tqdm.tqdm(enumerate(self.train_dataloader)):
                self.optimizer.zero_grad(set_to_none=True)            

                # TODO parse하는 부분 확인하기

                x = self.parse_sample(sample)
                mlm_labels = sample['mlm_labels'].to(self.device)
                _, mlm_output = self.model(x)
                loss = self.criterion(mlm_output.view(-1, 28996), mlm_labels.reshape(-1))

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
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': avg_mlm_loss,
                            'acc': avg_mlm_acc,
                            'epochs': n_epoch}, self.path+'_{}.pt'.format(n_epoch))
                print('Model parameter saved at epoch {}'.format(n_epoch))

            wandb.log({'epoch': n_epoch,
                       'train_mlm_loss:': avg_mlm_loss,
                       'train_mlm_acc': avg_mlm_acc
                       })

            print('[Train]  loss: {:.3f},  acc: {:.3f}'.format(avg_mlm_loss, avg_mlm_acc))


    def rnn_train(self):
        for n_epoch in range(self.n_epochs + 1):
            avg_mlm_loss = 0.0
            avg_mlm_acc = 0.0
            self.model.train()

            for iter, sample in tqdm.tqdm(enumerate(self.train_dataloader)):
                self.optimizer.zero_grad(set_to_none=True)

                input_id = sample['input'].to(self.device)
                mlm_labels = sample['mlm_labels'].to(self.device)

                _, mlm_output = self.model(input_id)

                loss = self.criterion(mlm_output.view(-1, 28996), mlm_labels.reshape(-1))
                avg_mlm_loss += loss.item() / len(self.train_dataloader)

                loss.backward()
                self.optimizer.step()

                # compute accuracy
                total_mlm_correct = 0
                total_mlm_element = 0

                eq = (mlm_output.argmax(dim=-1).eq(mlm_labels)).cpu().numpy()
                label_np = mlm_labels.cpu().numpy()

                for bs, label_i in enumerate(label_np):
                    index = np.where(label_i == -100)[0]
                    f_label = np.delete(label_i, index)
                    f_eq = np.delete(eq[bs], index)
                    total_mlm_correct += f_eq.sum()
                    total_mlm_element += len(f_label)

                acc_iter = total_mlm_correct / total_mlm_element
                avg_mlm_acc += acc_iter

            avg_mlm_acc /= len(self.train_dataloader)
            print(f'[MLM Loss]: {avg_mlm_loss}')
            print(f'[MLM Acc]:  {avg_mlm_acc}')

            wandb.log({'mlm_loss': avg_mlm_loss,
                       'mlm_acc': avg_mlm_acc})

            if n_epoch in [10, 25, 50, 100, 200, 400]:
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': avg_mlm_loss,
                            'acc': avg_mlm_acc,
                            'epochs': n_epoch}, self.path + '_{}.pt'.format(n_epoch))
                print(f'Model parameter saved at {n_epoch}')



