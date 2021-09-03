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
    parser.add_argument('--save_path', type=str, default='/home/ghhur/data/output/NIPS_output/checkpoints/')
    parser.add_argument('--model_path', type=str, default='/home/ghhur/data/output/NIPS_output/pretrain/pretrain.pt')

    # dataset
    parser.add_argument('--source_file', choices=['mimic', 'eicu', 'pooled'], type=str, default='mimic')
    parser.add_argument('--target_task', choices=['Readm', 'Mort', 'Los_3', 'Los_7', 'Dx'], type=str, default='Readm')

    # trainer
    parser.add_argument('--ratio', choices=['0', '10', '30', '50', '70', '90', '100'], type=str, default= None)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--value_embed_type', choices=['VA','DSVA','DSVA+DPE','VC'], default='VA')

    # enc model
    parser.add_argument('--text_encoder_model', choices=['bert', 'bio_clinical_bert', 'bio_bert', 'pubmed_bert', 'blue_bert', 'bert_mini', 'bert_tiny', 'bert_small', 'rnn'], type=str)
    parser.add_argument('--text_encoder_embed_dim', type=int, default=128)
    parser.add_argument('--text_encoder_hidden_dim', type=int, default=256)

    
    parser.add_argument('--rnn_layer', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0.3)
    parser.add_argument('--pred_embed_dim', type=int, default=128)
    parser.add_argument('--pred_hidden_dim ', type=int, default=256)

    # wnadb setting
    parser.add_argument('--wandb_project_name', type=str)
    parser.add_argument('--wandb_run_name', type=str)

    # pretrain & finetune
    parser.add_argument('--mlm_probability', type=float, default=0.0)
    parser.add_argument('--load_bert_scratch', action='store_true')
    parser.add_argument('--load_pretrained_weights', action='store_true')

    parser.add_argument('--transfer', type=str, 
                help = "example = {'lr':1e-4, 'epoch':400, 'src':'mimic', 'model':'bert_tiny'}") # should be finxed 
    
    # for transfer_fewshot
    parser.add_argument('--transfer_fewshot', action = 'store_true')
    parser.add_argument('--dest_file', choices = ['mimic', 'eicu'], default = 'eicu') 

    # mode
    parser.add_argument('--embed_model_mode', choices=['CodeEmb-RD', 'CodeEmb-W2V' 'BERT-CLS-FT', 'BERT-FT', 
            'BERT-Scr', 'BERT-FT+MLM', 'RNN-Scr', 'RNN-Scr+MLM', 'W2V-pretrain', 'MLM-pretrain-BERT', 'MLM-pretrain-RNN'], default='CodeEmb-RD')

    
    
    args = parser.parse_args()

    if (args.wandb_project_name is None) and (args.debug==False):
        raise AssertionError('wandb project name should not be null')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    SEED = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]

    print(f"Training Mode : {args.embed_model_mode}" )
    if args.embed_model_mode == 'W2V-pretrain':
        from trainers.Word2Vec_trainer import Trainer
        SEED = [2020]

    elif args.embed_model_mode in ['MLM-pretrain-BERT', 'MLM-pretrain-RNN']:
        from trainers.textencoder_bert_MLM_trainer import Trainer
        SEED = [2020]

    else:
        from trainers.base_trainer import Trainer


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
