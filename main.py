import torch
import torch.multiprocessing as mp

import random
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_number', type=str)

    # default setting
    parser.add_argument('--input_path', type=str, default='/home/ghhur/data/input/')
    parser.add_argument('--output_path', type=str, default='/home/ghhur/data/output/NIPS_output/')

    # dataset
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'both'], type=str, default='mimic')
    parser.add_argument('--target', choices=['Readm', 'Mort', 'Los_3', 'Los_7', 'Dx'], type=str, default='readmission')
    #parser.add_argument('--word_max_length', type=int, default=30)    -> preprocess 

    # trainer
    parser.add_argument('--ratio', choices=['0', '10', '30', '50', '70', '90', '100'], type=str, default= None)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--concat_type', choices=['VA','DSVA','DSVA+DPE','VC'], default='VA')
    #parser.add_argument('--loss_type', choices=['BCE','Focal'], default ='BCE')  -> BCE only

    # enc model
    parser.add_argument('--enc_model', choices=['bert', 'bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert', 'bert_mini', 'bert_tiny', 'bert_small', 'transformer', 'rnn'], type=str)
    parser.add_argument('--enc_rnn_model', choices=['gru', 'lstm'], type=str, default='gru')
    parser.add_argument('--enc_embed_dim', type=int, default=128)
    parser.add_argument('--enc_hidden_size', type=int, default=256)
    #parser.add_argument('--bert_hidden_from_cls', action='store_true')     -> always last hidden layer

    # pred model
    parser.add_argument('--pred_rnn_model', choices=['gru', 'lstm'], type=str, default='gru')
    
    parser.add_argument('--rnn_layer', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0.3)
    parser.add_argument('--pred_embed_dim', type=int, default=128)
    parser.add_argument('--pred_hidden_dim ', type=int, default=256)

    #parser.add_argument('--pred_model_mode', choices=['rnn','transformer'], default='rnn') -> rnn only
    #parser.add_argument('--rnn_att', choices=['no_att','single_att','multi_att'], type=str, default='no_att') -> no attention report
    #parser.add_argument('--RNN_SESTS', action='store_true') -> no report

    # wnadb setting
    parser.add_argument('--notes', type=str)
    parser.add_argument('--wandb_project_name', type=str)
    parser.add_argument('--wandb_run_name', type=str)

    # pretrain & finetune
    parser.add_argument('--mlm_probability', type=float, default=0.0)
    parser.add_argument('--load_bert_scratch', action='store_true')
    parser.add_argument('--transfer', type=str, 
                help = "example = {'lr':1e-4, 'epoch':400, 'src':'mimic', 'model':'bert_tiny'}") # should be finxed 
    
    # for transfer_fewshot
    parser.add_argument('--transfer_fewshot', action = 'store_true')
    parser.add_argument('--dest_file', choices = ['mimic', 'eicu'], default = 'eicu') 

    # mode
    parser.add_argument('--embed_model_mode', choices=['CodeEmb-RD', 'CodeEmb-W2V' 'BERT-CLS-FT', 'BERT-FT', 
            'BERT-Scr', 'BERT-FT+MLM', 'RNN-Scr', 'RNN-Scr+MLM', 'MLM-pretrain-BERT', 'MLM-pretrain-RNN'], default='CodeEmb-RD')
    parser.add_argument('--debug', action='store_true')
 
    
     
    # for trainer setting
    #parser.add_argument('--save_file', type=str)
    #parser.add_argument('--load_file', type=str)
    
    #parser.add_argument('--valid_epochs', type = int, default = 5)
    #parser.add_argument('--log_interval', type = int, default = 1)
    #parser.add_argument('--finetune', action = 'store_true')
    
    
    #parser.add_argument('--predictive_SSL', choices=['BYOL', 'BYOL_OT', 'SimCLR', 'Wav2Vec'], type=str)
    #parser.add_argument('--textencoder_ssl', choices=['reflect_freq', 'unique'], type=str, default=None)
    
    
    args = parser.parse_args()

    if (args.wandb_project_name is None) and (args.debug==False):
        raise AssertionError('wandb project name should not be null')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device =
   
    # predcitive model hyperparameter



    SEED = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]

    print(f"Training Mode : {args.embed_model_mode}" )
    if args.embed_model_mode != 'pretrain':
        if args.method == 'dp':
            from trainer.base_trainer import Trainer
        else:
            from trainer.ddp_trainer import DDPTrainer as Trainer


    if args.predictive_SSL == 'BYOL':
        from trainer.BYOL_trainer import BYOLTrainer as Trainer
        if args.transfer is None:
            SEED = [2020]

    elif args.predictive_SSL == 'SimCLR':
        from dataset.bert_pretraining_dataloader import bertinduced_get_dataloader as get_dataloader
        from trainer.SimCLR_trainer import SimCLR_Trainer as Trainer
        SEED = [2020]

    elif args.predictive_SSL == 'BYOL_OT':
        from trainer.BYOL_OT_trainer import BYOL_OT_Trainer as Trainer
        if args.transfer is None:
            SEED = [2020]

    if args.embed_model_mode == 'pretrain' and args.textencoder_mlm_probability > 0.0:
        from trainer.textencoder_bert_MLM_trainer import TextEncoderBert_MLM_Trainer as Trainer
        print('Bert_SSL {} {}'.format(args.bert_model, args.textencoder_ssl))
        SEED = [2020]

    elif args.predictive_SSL == 'Wav2Vec':
        from dataset.bert_pretraining_dataloader import bertinduced_get_dataloader as get_dataloader
        from trainer.wav2vec2_trainer import Wav2Vec2Trainer as Trainer
        SEED = [2020]

    if args.bert_model == 'rnn' and args.textencoder_mlm_probability > 0.0 and args.target == 'pretrain':
        from trainer.rnn_textencoder_trainer import RNNTextencoderTrainer as Trainer
        SEED = [2020]

    '''
    if args.Bert_SSL:
        from dataset.bert_pretraining_dataloader import bertinduced_get_dataloader as get_dataloader
        from trainer.bert_ssl_trainer import Bert_SSL_Trainer as Trainer
        print('Bert_SSL {}'.format(args.bert_model))

        SEED = [2020]
    '''

    # if args.time_window == '12':
    #     assert args.max_length == '150', "time_window of 12 should have max length of 150!"
    # elif args.time_window == '24':
    #     assert args.max_length == '200', "time_window of 24 should have max length of 200!"

    # if args.item == 'all':
    #     assert args.max_length == '300', 'when using all items, max length should `be` 300'

    mp.set_sharing_strategy('file_system')

    for seed in SEED:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True

        args.seed = seed
        print('seed_number', args.seed)

        trainer = Trainer(args, device)
        trainer.train()
                
        print('Finished training seed: {}'.format(seed))

if __name__ == '__main__':
    main()
