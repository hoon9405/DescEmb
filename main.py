import argparse
import logging
import logging.config
import random
import os
import sys

# should setup root logger before importing any relevant libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream = sys.stdout
)
logger = logging.getLogger(__name__)

import numpy as np

import torch
import torch.multiprocessing as mp

from trainers import Trainer, Word2VecTrainer

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--distributed_world_size', type=int, default=1
    )

    # checkpoint configs
    parser.add_argument('--input_path', type=str, default='/home/ghhur/data/input/')
    parser.add_argument('--model_path', type=str, default='/home/ghhur/data/output/NIPS_output/pretrain/pretrain.pt')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_prefix', type=str, default='checkpoint')

    parser.add_argument(
        '--patience', type=int, default=-1,
        help= (
            'early stop training if valid performance does not '
            + 'improve for N consecutive validation runs'
        )
    )
    parser.add_argument(
        '--disable_validation', action='store_true',
        help='disable validation'
    )

    # dataset
    parser.add_argument('--data', choices=['mimic', 'eicu', 'pooled'], type=str, required=True)
    parser.add_argument('--eval_data', choices=['mimic', 'eicu', 'pooled'], type=str, default=None, required=False)
    parser.add_argument('--value_embed_type', choices=['VA','DSVA','DSVA_DPE','VC', 'nonconcat'], default='VA')
    parser.add_argument('--fold', type=str, default=None)
    parser.add_argument('--valid_subsets', type=str, default="valid, test")

    parser.add_argument(
        '--task',
        choices=['mlm', 'w2v', 'readmission', 'mortality', 'los_3day', 'los_7day', 'diagnosis'],
        type=str,
        default='readmission',
        help=""
    )

    # trainer
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--ratio', choices=['10', '30', '50', '70', '90', '100'], type=str, default= None)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    
    # encoder model configs
    parser.add_argument('--enc_embed_dim', type=int, default=128)
    parser.add_argument('--enc_hidden_dim', type=int, default=256)

    # predictive model configs
    parser.add_argument('--rnn_layer', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0.3)
    parser.add_argument('--pred_embed_dim', type=int, default=128)
    parser.add_argument('--pred_hidden_dim', type=int, default=256)

    # wandb setting
    # parser.add_argument('--wandb_project_name', type=str)
    # parser.add_argument('--wandb_run_name', type=str)

    # mlm pretrain & finetune
    parser.add_argument('--mlm_prob', type=float, default=0.3)
    parser.add_argument('--load_pretrained_weights', action='store_true')

    # parser.add_argument('--transfer', type=str, 
    #             help = "example = {'lr':1e-4, 'epoch':400, 'src':'mimic', 'model':'bert_tiny'}") # should be finxed 
    
    # for transfer
    parser.add_argument("--transfer", action="store_true")
    # parser.add_argument('--dest_file', choices = ['mimic', 'eicu'], default = 'eicu') 

    # model
    parser.add_argument(
        '--model', type=str, required=True,
        help='name of the model to be trained'
    )
    parser.add_argument(
        '--embed_model', type=str, required=False,
        help='name of the encoder model in the --model, '
            'only used when --model has encoder-predictor structure'
    )
    parser.add_argument(
        '--pred_model', type=str, required=False,
        help='name of the predictor model in the --model, '
            'only used when --model has encoder-predictor structure'
    )

    parser.add_argument('--bert_model', choices=['bert', 'bert_tiny', 'bert_mini'], type=str, default='bert_tiny')
    parser.add_argument('--init_bert_params', action='store_true')
    parser.add_argument('--init_bert_params_with_freeze', action='store_true')

    # parser.add_argument('--embed_model_mode', choices=['CodeEmb-RD', 'CodeEmb-W2V' 'BERT-CLS-FT', 'BERT-FT', 
    #         'BERT-Scr', 'BERT-FT+MLM', 'RNN-Scr', 'RNN-Scr+MLM', 'W2V-pretrain', 'MLM-pretrain-BERT', 'MLM-pretrain-RNN'], default='CodeEmb-RD')

    return parser

def main():
    args = get_parser().parse_args()
    args.valid_subsets = (
        args.valid_subsets.replace(' ','').split(',')
        if (
            not args.disable_validation
            and args.task not in ['mlm', 'w2v']
            and args.valid_subsets
        )
        else []
    )
    args.device_ids = list(range(args.distributed_world_size))
    set_struct(vars(args))

    # print(f"Training Mode : {args.embed_model_mode}" )
    # if args.embed_model_mode == 'W2V-pretrain':
    #     from trainers.Word2Vec_trainer import Trainer
    #     SEED = [2020]

    # elif args.embed_model_mode in ['MLM-pretrain-BERT', 'MLM-pretrain-RNN']:
    #     from trainers.textencoder_bert_MLM_trainer import Trainer
    #     SEED = [2020]
    # else:
    #     from trainers.base_trainer import Trainer


    mp.set_sharing_strategy('file_system')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True

    trainer = Trainer(args)
    trainer.train()
    logger.info("done training")


def set_struct(cfg: dict):
    root = os.path.abspath(
        os.path.dirname(__file__)
    )
    from datetime import datetime
    now = datetime.now()
    from pytz import timezone
    # apply timezone manually
    now = now.astimezone(timezone('Asia/Seoul'))

    output_dir = os.path.join(
        root,
        "outputs",
        now.strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S")
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(output_dir)

    job_logging_cfg = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'train.log'
            }
        },
        'root': {
            'level': 'INFO', 'handlers': ['console', 'file']
            },
        'disable_existing_loggers': False
    }
    logging.config.dictConfig(job_logging_cfg)

    cfg_dir = ".config"
    os.mkdir(cfg_dir)
    os.mkdir(cfg['save_dir'])

    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        for k, v in cfg.items():
            print("{}: {}".format(k, v), file=f)


if __name__ == '__main__':
    main()
