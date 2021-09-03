import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import os
import tqdm
from torch.utils.data import DataLoader
from models.rnn_models import *
from utils.loss import *
from utils.trainer_utils import *

from datasets import (
  Dataset,
  TokenizedDataset
)

from models.EHRmodel import EHRModel
from models.layers import Code_Inputlayer, Desc_InputLayer
from models.textencoder_bert import TextEncoder_BERT
from models.textencoder_rnn import TextEncoder_RNN
from models.rnn_models import RNNModels


class Trainer:
    def __init__(self, args, device):
        self.seed = args.seed
        self.path = args.path
        self.visualize = args.visualize
        self.device = device
        self.debug = args.debug
        self.source_file = args.source_file
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.target = args.target
        self.embed_model_mode = args.embed_model_mode
        self.pred_model_mode = args.pred_model_mode
        self.mlm_prob = args.textencoder_mlm_probability
        self.concat_type = args.concat_type
        self.ratio = args.ratio

        if args.embed_model_mode in ['CodeEmb']:
          args.embed_model = Code_Inputlayer
          self.dataset = Dataset
        
        elif args.embed_model_mode in ['DescEmb-FR', 'DescEmb-FT']:
          args.embed_model = Desc_InputLayer
          self.dataset = Dataset
        
        elif args.embed_model_mode in ['Scratch-RNN']:
          args.embed_model = TextEncoder_RNN
          self.dataset = TokenizedDataset
        
        elif args.embed_model_mode in ['Scratch-Bert', 'Finetune-Bert']:
          args.embed_model = TextEncoder_BERT
          self.dataset = TokenizedDataset

        if args.pred_model_mode =='rnn':
          args.pred_model = RNNModels
        
        elif args.pred_model_mode =='transformer':
          args.pred_model = Transformer

        if not self.debug:
            wandb.init(project=args.wandb_project_name, entity="pretrained_ehr", config=args, reinit=True)

            print('wandb project name  : ', args.wandb_project_name)  
        
        if args.pred_model_mode =='rnn':     # file name for singleRNN
          if args.Word2Vec:
            filename = 'rnn_Word2Vec_{}_{}_{}_layers{}_hidden{}_{}_{}'.format(args.bert_model, args.rnn_type, args.rnn_att,
                                                                     args.rnn_layer, args.pred_hidden_dim, args.loss_type, args.seed)

          if args.finetune is True:
            filename = 'rnn_finetune_{}_{}_{}_layers{}_hidden{}_{}_{}'.format(args.bert_model, args.rnn_type, args.rnn_att, args.rnn_layer,
                                                                              args.pred_hidden_dim, args.loss_type, args.seed)

          else:
            filename = 'rnn_{}_{}_{}_layers{}_hidden{}_{}_{}'.format(args.bert_model, args.rnn_type, args.rnn_att,
                                                      args.rnn_layer, args.pred_hidden_dim, args.loss_type, args.seed)
      

        elif args.pred_model_mode =='transformer':
          filename = 'transformer_{}_{}_layers{}_attnheads{}_hidden{}_{}_{}'.format(args.bert_model, 
                                      args.transformer_att, args.transformer_layers, args.transformer_attn_heads, 
                                      args.pred_hidden_dim, args.loss_type, args.seed)

        if args.concat_type !='nonconcat':
          filename += '_' + str(args.concat_type)

        if args.textencoder_mlm_probability !=0.0:
          filename += '_' + 'textencoder_mlm' + '_' + str(args.textencoder_mlm_probability)

        if not args.transfer_fewshot and not args.ratio is None:
               filename = filename + '_' + args.ratio
        path = os.path.join(args.path, args.embed_model_mode, args.source_file, args.target, filename)
        
        print('Embed_model_mode : {},  Pred_model_mode : {} '.format(args.embed_model_mode, args.pred_model_mode))
        print('Embedding_dim size for pred model: {}'.format(args.pred_embed_dim))
        print('Model will be saved in {}'.format(path))

        self.best_eval_path = path + '_best_auprc.pt'
        self.best_mimic_eval_path = path + '_mimic_best_auprc.pt'
        self.best_eicu_eval_path = path + '_eicu_best_auprc.pt'

        tmp = None 
        #Model define
        if args.transfer_fewshot is True:
          filename += '_' + args.ratio
          path = os.path.join(args.path, args.embed_model_mode, f"{args.source_file}2{args.dest_file}")
          tmp = args.source_file
          args.source_file = args.dest_file

          if not os.path.exists(path):
              os.mkdir(path)
          if not os.path.exists(os.path.join(path, 'readmission')):
              os.mkdir(os.path.join(path, 'readmission'))
          if not os.path.exists(os.path.join(path, 'mortality')):
              os.mkdir(os.path.join(path, 'mortality'))
          if not os.path.exists(os.path.join(path, 'los_3day')):
              os.mkdir(os.path.join(path, 'los_3day'))
          if not os.path.exists(os.path.join(path, 'los_7day')):
              os.mkdir(os.path.join(path, 'los_7day'))
          if not os.path.exists(os.path.join(path, 'diagnosis')):
              os.mkdir(os.path.join(path, 'diagnosis'))
          path = os.path.join(path, args.target, filename)
          
          ehr_model = EHRModel(args)
          print('load model from ..', self.best_eval_path)
          state_dict = torch.load(self.best_eval_path)['model_state_dict']
          if args.embed_model_mode in ['CodeEmb', 'DescEmb-FR', 'DescEmb-FT']:
              state_dict = {k.replace('module.',''): state_dict[k] for k in state_dict if (not 'embedding' in k and not 'bert_embed' in k)}
          else:
              state_dict = {k.replace('module.',''): state_dict[k] for k in state_dict }
          ehr_model.load_state_dict(state_dict, strict = False)

          if args.ratio != '0':
            self.best_eval_path = path + '_best_auprc.pt'
            self.best_mimic_eval_path = path + '_mimic_best_auprc.pt'
            self.best_eicu_eval_path = path + '_eicu_best_auprc.pt'
        else:
          ehr_model = EHRModel(args)

        self.model = nn.DataParallel(ehr_model).to(self.device)
       
        #Dataloader define
        self.train_dataloader = None
        self.eval_dataloader = None
        self.test_dataloader = None

        if self.visualize == True:
            args.source_file = 'mimic'
            self.total_mimic_dataloader = DataLoader(self.dataset(args, 'test'), batch_size = args.batch_size, num_workers=8, shuffle = False)
            args.source_file = 'eicu'
            self.total_eicu_dataloader = DataLoader(self.dataset(args, 'test'), batch_size = args.batch_size, num_workers=8, shuffle = False)
            args.source_file = 'both'

        if args.ratio != '0':
          self.train_dataloader = DataLoader(self.dataset(args, 'train'), batch_size = args.batch_size, num_workers=8, shuffle = True)
        if not args.source_file == 'both':
          self.eval_dataloader = DataLoader(self.dataset(args, 'valid'), batch_size = args.batch_size, num_workers=8, shuffle = True)
          self.test_dataloader = DataLoader(self.dataset(args, 'test'), batch_size = args.batch_size, shuffle = False)
        else:
          tmp = args.source_file
          args.source_file = 'both_mimic'
          self.mimic_eval_dataloader = DataLoader(self.dataset(args, 'valid'), batch_size = args.batch_size, num_workers=8, shuffle = True)
          self.mimic_test_dataloader = DataLoader(self.dataset(args, 'test'), batch_size = args.batch_size, shuffle = False)
          args.source_file = 'both_eicu'
          self.eicu_eval_dataloader = DataLoader(self.dataset(args, 'valid'), batch_size = args.batch_size, num_workers=8, shuffle = True)
          self.eicu_test_dataloader = DataLoader(self.dataset(args, 'test'), batch_size = args.batch_size, shuffle = False)
        
        if tmp is not None:
          args.source_file = tmp
          
        if args.loss_type =='BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss_type =='Focal':
            self.criterion = FocalLoss()      
      
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        print('train dataloader total data number ', len(self.train_dataloader)*args.batch_size if self.train_dataloader else 0)
        if self.source_file != 'both':
            self.early_stopping = EarlyStopping(patience=20, verbose=True)
        
            print('eval dataloader total data number ', len(self.eval_dataloader)*args.batch_size)
            print('test dataloader total data number ', len(self.test_dataloader)*args.batch_size)
        
        elif self.source_file == 'both':
            self.mimic_early_stopping = EarlyStopping(patience=20, verbose=True)
            self.eicu_early_stopping = EarlyStopping(patience=20, verbose=True)
            
            print('mimic_eval dataloader total data number ', len(self.mimic_eval_dataloader)*args.batch_size)
            print('eicu_eval dataloader total data number ', len(self.eicu_eval_dataloader)*args.batch_size)
            print('mimic_test dataloader total data number ', len(self.mimic_test_dataloader)*args.batch_size)
            print('eicu_test dataloader total data number ', len(self.eicu_test_dataloader)*args.batch_size)
    
        num_params = count_parameters(self.model)
        print('Number of parameters: {}'.format(num_params))

        #Sample to inpur convert 
    def parse_sample(self, sample):
      if self.embed_model_mode in ['Scratch-Bert', 'Finetune-Bert']:
        x={}
        x['input_ids'] = sample['input'].to(self.device)
        x['token_type_ids'] = sample['token_type_ids'].to(self.device)
        x['attention_mask'] = sample['attention_mask'].to(self.device)

      elif self.embed_model_mode =='Scratch-RNN' and self.concat_type =='concat_c':
        x={} 
        x['input_ids'] = sample['input'].to(self.device)
        x['token_type_ids'] = sample['token_type_ids'].to(self.device)
        
      else:
        x = sample['input'].to(self.device)
      
      if self.pred_model_mode =='transformer':
        var = sample['offset_order'].to(self.device)
      
      elif self.pred_model_mode == 'rnn':
        var = sample['seq_len'].to(self.device)

      return x, var
      
    def train(self):
        break_token = False
        best_auprc = 0.0
        best_mimic_auprc = 0.0
        best_eicu_auprc = 0.0

        if self.train_dataloader is not None:
          for n_epoch in range(self.n_epochs + 1):

            preds_train = []
            truths_train = []
            avg_train_loss = 0.
            
            self.model.train()

            for iter, sample in tqdm.tqdm(enumerate(self.train_dataloader)):
              self.optimizer.zero_grad(set_to_none=True)
              label = sample.pop('labels').to(self.device)              
              value = sample.pop('value').to(self.device)
             
              x, var = self.parse_sample(sample)

              if self.mlm_prob > 0:
                y_pred, mlm_output = self.model(x, var, value)
              else:
                y_pred = self.model(x, var, value)
          
              if self.target == 'diagnosis':
                loss = self.criterion(y_pred, label.squeeze(2).float())
              else:
                loss = self.criterion(y_pred, label.float())

              if self.mlm_prob > 0:
                mlm_labels = sample['mlm_labels'].to(self.device)
                mlm_labels = mlm_labels.view(-1)
                mlm_output = mlm_output.view(-1, 28996)

                survivor = torch.where(mlm_labels != -100)[0]

                mlm_labels = mlm_labels[survivor]
                mlm_output = mlm_output[survivor]

                extra_loss = F.cross_entropy(mlm_output, mlm_labels)
                loss += extra_loss

              loss.backward()
              self.optimizer.step()

              avg_train_loss += loss.item() / len(self.train_dataloader)

              probs_train = torch.sigmoid(y_pred).detach().cpu().numpy()
              preds_train += list(probs_train.flatten())
              truths_train += list(label.detach().cpu().numpy().flatten())         

            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train, average='micro')

            if not self.debug:
              wandb.log({'train_loss': avg_train_loss,
                      'train_auroc': auroc_train,
                      'train_auprc': auprc_train})
            print('[Train]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_train_loss, auroc_train, auprc_train))

            if self.source_file != 'both':
              break_token, best_auprc = self.evaluation(best_auprc, n_epoch)            

            elif self.source_file == 'both':
              break_token, best_mimic_auprc, best_eicu_auprc = self.evaluation_both(best_mimic_auprc, best_eicu_auprc, n_epoch)
            
            if break_token: break

        if self.source_file != 'both':
            self.test()
        elif self.source_file == 'both':
            self.test_both()

     
    def inference(self, dataloader=None, mode='eval'):
        self.model.eval()

        preds_eval = []
        truths_eval = []
        avg_eval_loss = 0.

        with torch.no_grad():
            for iter, sample in enumerate(dataloader):
              label = sample.pop('labels').to(self.device)
              value = sample.pop('value').to(self.device)
              x, var = self.parse_sample(sample)

              if self.mlm_prob > 0.:
                y_pred, mlm_output = self.model(x, var, value)
              else:
                y_pred = self.model(x, var, value)

              if self.target == 'diagnosis':
                  loss = self.criterion(y_pred, label.squeeze(2).float())
              else:
                  loss = self.criterion(y_pred, label.float())

              avg_eval_loss += loss.item() / len(dataloader)

              probs_eval = torch.sigmoid(y_pred).detach().cpu().numpy()
              preds_eval += list(probs_eval.flatten())
              truths_eval += list(label.detach().cpu().numpy().flatten())

            auroc_eval = roc_auc_score(truths_eval, preds_eval)
            auprc_eval = average_precision_score(truths_eval, preds_eval, average='micro')

        return avg_eval_loss, auroc_eval, auprc_eval


    def evaluation(self, best_auprc, n_epoch):
      break_token = False
      avg_eval_loss, auroc_eval, auprc_eval = self.inference(dataloader=self.eval_dataloader, mode='eval')

      if not self.debug:
        wandb.log({'eval_loss': avg_eval_loss,
                      'eval_auroc': auroc_eval,
                      'eval_auprc': auprc_eval})
                      
        if best_auprc < auprc_eval:
            best_loss = avg_eval_loss
            best_auroc = auroc_eval
            best_auprc = auprc_eval
            print('[Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(avg_eval_loss, auroc_eval, auprc_eval))

            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_loss,
                        'auroc': best_auroc,
                        'auprc': best_auprc,
                        'epochs': n_epoch}, self.best_eval_path)

            print('Model parameter saved at epoch {}'.format(n_epoch))

      self.early_stopping(auprc_eval)
      if self.early_stopping.early_stop:
        print('Early stopping')
        break_token = True

      return break_token, best_auprc

    def evaluation_both(self, best_mimic_auprc, best_eicu_auprc, n_epoch):
      break_token = False
      mimic_avg_eval_loss, mimic_auroc_eval, mimic_auprc_eval = self.inference(dataloader=self.mimic_eval_dataloader, mode='eval')
      eicu_avg_eval_loss, eicu_auroc_eval, eicu_auprc_eval = self.inference(dataloader=self.eicu_eval_dataloader, mode='eval')

      if not self.debug:
        wandb.log({'mimic_eval_loss': mimic_avg_eval_loss,
                    'mimic_eval_auroc': mimic_auroc_eval,
                    'mimic_eval_auprc': mimic_auprc_eval,
                    'eicu_eval_loss': eicu_avg_eval_loss,
                    'eicu_eval_auroc': eicu_auprc_eval,
                    'eicu_eval_auprc': eicu_auprc_eval})

        if best_mimic_auprc < mimic_auprc_eval:
          best_mimic_loss = mimic_avg_eval_loss
          best_mimic_auroc = mimic_auroc_eval
          best_mimic_auprc = mimic_auprc_eval
      
          torch.save({'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': best_mimic_loss,
                      'auroc': best_mimic_auroc,
                      'auprc': best_mimic_auprc,
                      'epochs': n_epoch}, self.best_mimic_eval_path)
          print('[mimic] Model parameter saved at epoch {}'.format(n_epoch))

        if best_eicu_auprc < eicu_auprc_eval:
            best_eicu_loss = eicu_avg_eval_loss
            best_eicu_auroc = eicu_auroc_eval
            best_eicu_auprc = eicu_auprc_eval
      
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_eicu_loss,
                        'auroc': best_eicu_auroc,
                        'auprc': best_eicu_auprc,
                        'epochs': n_epoch}, self.best_eicu_eval_path)
            print('[eicu] Model parameter saved at epoch {}'.format(n_epoch))

      print('[mimic/Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(mimic_avg_eval_loss, mimic_auroc_eval, mimic_auprc_eval))
      print('[eicu/Valid]  loss: {:.3f},  auroc: {:.3f},   auprc: {:.3f}'.format(eicu_avg_eval_loss, eicu_auroc_eval, eicu_auprc_eval))

      self.mimic_early_stopping(mimic_auprc_eval)
      self.eicu_early_stopping(eicu_auprc_eval)

      if self.mimic_early_stopping.early_stop and self.eicu_early_stopping.early_stop:
        print('Early Stopping')
        break_token = True
      
      return break_token, best_mimic_auprc, best_eicu_auprc


    def test(self):
      print('test!')
      state_dict = torch.load(self.best_eval_path)['model_state_dict']
      if self.ratio == '0':
        state_dict = {k.replace('module.',''): state_dict[k] for k in state_dict if (not 'embedding' in k and not 'bert_embed' in k)}
      self.model.load_state_dict(state_dict, strict = False)

      avg_test_loss, auroc_test, auprc_test = self.inference(dataloader=self.test_dataloader, mode='test')
      
      if not self.debug:
          wandb.log({'test_loss': avg_test_loss,
                      'test_auroc': auroc_test,
                      'test_auprc': auprc_test})

      print('[Test]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))
    
    def test_both(self):
        assert self.ratio != '0', "not yet implemented for ratio 0 with pooled dataset."

        ckpt = torch.load(self.best_mimic_eval_path)
        print('best_mimic model path ', self.best_mimic_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        avg_test_loss, auroc_test, auprc_test = self.inference(dataloader=self.mimic_test_dataloader, mode='test')          

        if not self.debug:
            wandb.log({'mimic_test_loss': avg_test_loss,
                        'mimic_test_auroc': auroc_test,
                        'mimic_test_auprc': auprc_test})

        print('[Test/mimic]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test,
                                                                                    auprc_test))

        ckpt = torch.load(self.best_eicu_eval_path)
        print('best_eicu model path ', self.best_eicu_eval_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        avg_test_loss, auroc_test, auprc_test = self.inference(dataloader=self.eicu_test_dataloader, mode='test')   

        if not self.debug:
            wandb.log({'eicu_test_loss': avg_test_loss,
                        'eicu_test_auroc': auroc_test,
                        'eicu_test_auprc': auprc_test})

        print('[Test/eicu]  loss: {:.3f},     auroc: {:.3f},     auprc:   {:.3f}'.format(avg_test_loss, auroc_test, auprc_test))

    def test_for_visualize(self):
        target =self.target
        embed_model_mode = self.embed_model_mode
        df = pd.DataFrame([], columns = ['y_pred','label'])
        
        ckpt = torch.load(self.best_mimic_eval_path)
        print('best_mimic model path ', self.best_mimic_eval_path)
        print('n_epoch', ckpt['epochs'])
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)

        self.model.eval()

        y_pred_list = []
        y_label_list = []
        
        with torch.no_grad():
            for iter, sample in tqdm.tqdm(enumerate(self.total_mimic_dataloader)):
              label = sample.pop('labels').to(self.device)
              value = sample.pop('value').to(self.device)
              x, var = self.parse_sample(sample)
              y_pred = self.model(x, var, value)

              y_pred_list.extend(y_pred.tolist())
              y_label_list.extend(label.detach().cpu().numpy().flatten())

        df['y_pred'] = pd.Series(y_pred_list)
        df['label'] = pd.Series(y_label_list)
        
        #save mimic inference
        
        save_mimic_path = self.path + 'mimic_model_mimic_{}_{}_{}.pkl'.format(target, embed_model_mode, self.seed)

        df.to_pickle(save_mimic_path)

        df = pd.DataFrame([], columns = ['y_pred','label'])

        y_pred_list = []
        y_label_list = []
        
        with torch.no_grad():
            for iter, sample in tqdm.tqdm(enumerate(self.total_eicu_dataloader)):
              label = sample.pop('labels').to(self.device)
              value = sample.pop('value').to(self.device)
              x, var = self.parse_sample(sample)
              y_pred = self.model(x, var, value)

              y_pred_list.extend(y_pred.tolist())
              y_label_list.extend(label.detach().cpu().numpy().flatten())

      
        df['y_pred'] = pd.Series(y_pred_list)
        df['label'] = pd.Series(y_label_list)

        #save mimic inference

        save_eicu_path = self.path + 'mimic_model_eicu_{}_{}_{}.pkl'.format(target, embed_model_mode, self.seed)

        df.to_pickle(save_eicu_path)
        

      


