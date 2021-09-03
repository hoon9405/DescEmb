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
        self.path = args.pretrain_path
        print('Model will be saved in {}'.format(self.path))
        
        self.dataset = TokenizedDataset
        self.train_dataloader = DataLoader(self.dataset(args, 'train'), batch_size = args.batch_size, num_workers=8, shuffle = True)

        # For BERTs
        if self.embed_model_mode == 'MLM-pretrain-BERT':
            self.model = TextEncoder_BERT(args).cuda()

        elif self.embed_model_mode == 'MLM-pretrain-RNN':
            self.model = TextEncoder_RNN(args).to(device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        num_params = count_parameters(self.model)
        print('Number of parameters: {}'.format(num_params))


    def parse_sample(self, sample):
        if self.embed_model_mode == 'MLM-pretrain-BERT':
            x = {}
            x['input_ids'] = sample['input'].to(self.device)
            x['token_type_ids'] = sample['token_type_ids'].to(self.device)
            x['attention_mask'] = sample['attention_mask'].to(self.device)
            return x
        elif self.embed_model_mode == 'MLM-pretrain-RNN':
            return sample['input'].to(self.device)


    def train(self):
        for n_epoch in range(self.n_epochs + 1):
            avg_mlm_acc = 0.0
            avg_mlm_loss = 0.0
            self.model.train()

            for iter, sample in tqdm.tqdm(enumerate(self.train_dataloader)):
                self.optimizer.zero_grad(set_to_none=True)

                x = self.parse_sample(sample)
                mlm_labels = sample['mlm_labels'].to(self.device)

                _, mlm_output = self.model.mlm_forward(x)

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

                if total_mlm_element == 0 or type(total_mlm_element) != int:
                    print('total_mlm_element',total_mlm_element)
                    print('index', index)
                    print('f_label', f_label)

                acc_iter = total_mlm_correct / total_mlm_element
                avg_mlm_acc += acc_iter / len(self.train_dataloader)


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
