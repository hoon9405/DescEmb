import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


from dataset.dataset import TokenizedDataset
from models.textencoder_rnn import TextEncoder_RNN
from utils.trainer_utils import EarlyStopping

import wandb
import os
import tqdm


class RNNTextencoderTrainer():
    def __init__(self, args, device):
        if args.target == 'pretrain':
            self.dataloader = DataLoader(TokenizedDataset(args, data_type='train'), batch_size=args.batch_size, shuffle=True, num_workers=16)
        else:
            self.dataloader = DataLoader(TokenizedDataset(args, data_type='train'), batch_size=args.batch_size, shuffle=True, num_workers=16)
            self.valid_dataloader = DataLoader(TokenizedDataset(args, data_type='eval'), batch_size=args.batch_size, shuffle=True, num_workers=16)
            self.test_dataloader = DataLoader(TokenizedDataset(args, data_type='test'), batch_size=args.batch_size, shuffle=True, num_workers=16)
            self.early_stopping = EarlyStopping(patience=20, verbose=True)

        self.n_epochs = args.n_epochs
        self.device = device
        self.debug = args.debug
        self.source_file = args.source_file
        self.target = args.target
        self.textencoder_mlm_probability = args.textencoder_mlm_probability

        if (args.loss_type == 'Focal'):
            raise NotImplementedError

        if not args.debug:
            wandb.init(project='rnn_textencoder_mlm', entity='pretrained_ehr', config=args, reinit=True)

        if args.target == 'pretrain':
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        if args.target == 'pretrain':
            filename = 'mlm_predictive_rnn_{}_{}_{}_{}'.format(args.source_file, args.bert_model, args.textencoder_ssl, args.textencoder_mlm_probability)
            path = os.path.join(args.path, 'Scratch-RNN', filename)
        else:
            filename = f'predictive_rnn_{args.bert_model}_{args.lr}_{args.textencoder_ssl}_{args.seed}'
            path = os.path.join(args.path, 'SSL', 'MLM', args.source_file, args.target, filename)
        print(f'Model will be saved in {path}')

        self.best_eval_path = path + '_best_eval.pt'
        self.path = path

        # bring the model
        if args.target == 'pretrain':
            self.model = TextEncoder_RNN(args).to(device)
        else:
            self.model = RNNTransformerModel(args, device).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

    def train(self):
        if self.textencoder_mlm_probability > 0.:
            self.mlm_train()
        else:
            self.finetune_train()

    def mlm_train(self):
        for n_epoch in range(self.n_epochs + 1):
            avg_mlm_loss = 0.0
            avg_acc = 0.0
            self.model.train()

            for iter, sample in tqdm.tqdm(enumerate(self.dataloader)):
                self.optimizer.zero_grad(set_to_none=True)

                input_ids = sample['input'].to(self.device)
                attn_mask = sample['attention_mask'].to(self.device)
                mlm_labels = sample['mlm_labels'].to(self.device)

                _, mlm_output = self.model(input_ids)    # (B, S, 30000)

                # attn_mask = attn_mask.reshape(-1, input_ids.size(-1))
                # max_len = attn_mask.sum(1).max()
                # mlm_labels = mlm_labels[..., :max_len]

                loss = self.criterion(mlm_output.view(-1, 28996), mlm_labels.reshape(-1))
                avg_mlm_loss += loss.item() / len(self.dataloader)

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

                # if total_mlm_element == 0 or type(total_mlm_element) != int:
                #     print('total_mlm_element',total_mlm_element)
                #     print('index', index)
                #     print('f_label', f_label)

                acc_iter = total_mlm_correct / total_mlm_element
                avg_acc += acc_iter

            avg_acc /= len(self.dataloader)
            print(f'[MLM Loss]: {avg_mlm_loss}')
            print(f'[MLM Acc]:  {avg_acc}')

            wandb.log({'mlm_loss': avg_mlm_loss,
                       'mlm_acc': avg_acc})

            if n_epoch in [10, 25, 50, 100, 200, 400]:
                if not self.debug:
                    torch.save({'rnntextencoder_state_dict': self.model.state_dict()}, self.path + f'_{n_epoch}.pt')
                    print(f'Model parameter saved at {n_epoch}')

    def finetune_train(self):
        best_auprc = 0.
        for n_epoch in range(self.n_epochs+1):
            preds_train = []
            truths_train = []
            avg_train_loss = 0.
            self.model.train()

            for iter, sample in tqdm.tqdm(enumerate(self.dataloader)):
                self.optimizer.zero_grad(set_to_none=True)

                input_id = sample['input'].to(self.device)
                # token_type_id = sample['token_type_id'].to(self.device)
                attn_mask = sample['attention_mask'].to(self.device)
                offset_order = sample['offset_order'].to(self.device)
                label = sample['labels'].to(self.device)

                y_pred = self.model(input_id, attn_mask, offset_order)

                if self.target == 'diagnosis':
                    loss = nn.BCEWithLogitsLoss()(y_pred, label.float())
                else:
                    loss = nn.BCEWithLogitsLoss()(y_pred, label.squeeze().float())

                loss.backward()
                self.optimizer.step()

                avg_train_loss += loss.item() / len(self.dataloader)

                probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(label.detach().cpu().numpy().flatten())

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train, average='micro')

            avg_eval_loss, auroc_eval, auprc_eval = self.finetune_evaluation()

            print('[Train]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))
            print('[Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_eval_loss, auroc_eval, auprc_eval))

            if not self.debug:
                wandb.log({'train_loss': avg_train_loss,
                           'train_auroc': auroc_train,
                           'train_auprc': auprc_train,
                           'eval_loss': avg_eval_loss,
                           'eval_auroc': auroc_eval,
                           'eval_auprc': auprc_eval})

            if best_auprc < auprc_eval:
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict()}, self.best_eval_path)

            self.early_stopping(auprc_eval)
            if self.early_stopping.early_stop:
                break

        self.finetune_test()

    def finetune_evaluation(self):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.

        with torch.no_grad():
            for iter, sample in tqdm.tqdm(enumerate(self.valid_dataloader)):
                input_id = sample['input'].to(self.device)
                token_type_id = sample['token_type_id'].to(self.device)
                attn_mask = sample['attn_mask'].to(self.device)
                offset_order = sample['offset_order'].to(self.device)
                label = sample['labels'].to(self.device)

                y_pred = self.model(input_id, attn_mask, offset_order)

                if self.target != 'diagnosis':
                    loss = nn.BCEWithLogitsLoss()(y_pred, label.float())
                else:
                    loss = nn.BCEWithLogitsLoss()(y_pred, label.squeeze().float())

                avg_eval_loss += loss.item() / len(self.valid_dataloader)

                probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_eval += list(probs_eval.flatten())
                truths_eval += list(label.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

        return avg_eval_loss, auroc_eval, auprc_eval


    def finetune_test(self):
        ckpt = torch.load(self.best_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])

        self.model.eval()

        preds_test = []
        truths_test = []
        avg_test_loss = 0.

        with torch.no_grad():
            for iter, sample in tqdm.tqdm(enumerate(self.test_dataloader)):
                input_id = sample['input'].to(self.device)
                token_type_id = sample['token_type_id'].to(self.device)
                attn_mask = sample['attn_mask'].to(self.device)
                offset_order = sample['offset_order'].to(self.device)
                label = sample['labels'].to(self.device)

                y_pred = self.model(input_id, attn_mask, offset_order)

                if self.target != 'diagnosis':
                    loss = nn.BCEWithLogitsLoss()(y_pred, label.float())
                else:
                    loss = nn.BCEWithLogitsLoss()(y_pred, label.squeeze().float())

                avg_test_loss += loss.item() / len(self.test_dataloader)

                probs_test = torch.sigmoid(y_pred).detach().cpu().numpy()
                preds_test += list(probs_test.flatten())
                truths_test += list(label.detach().cpu().numpy().flatten())

            auroc_test = roc_auc_score(truths_test, preds_test)
            auprc_test = average_precision_score(truths_test, preds_test, average='micro')

        wandb.log({'test_loss': avg_test_loss,
                   'test_auroc': auroc_test,
                   'test_auprc': auprc_test})

        print('[Test]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))








