import numpy as np
import torch
import os
from collections import OrderedDict

'''
reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
modified for our purpose
'''

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 0
        self.delta = delta
        #self.path = path
        self.trace_func = trace_func
    def __call__(self, val_auroc):

        score = val_auroc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                self.trace_func(f'Validation AUROC increased ({self.val_loss_min:.6f} --> {val_auroc:.6f})')
            self.val_loss_min = val_auroc
            self.counter = 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def text_encoder_load_path(args):
    """Text encoder part model loading"""
    path = args.path
    print('path', path)

    transfer=args.transfer
    pretrain_lr = transfer['lr']
    pretrain_epoch = transfer['epoch']
    pretrain_src = transfer['src']
    pretrain_model = transfer['model']
    pretrain_concat_type = transfer['concat_type']
    file_path = os.path.join(path, 'SSL', 'MLM', 'pretrain')
    
    print('file_path', file_path)
    file_name = '{}_{}_{}_{}_None_{}_{}_concat_type_{}_{}.pt'.format(pretrain_src, pretrain_model,
                   pretrain_lr, '256', args.textencoder_mlm_scratch, 'False', pretrain_concat_type, pretrain_epoch)
    
    model_path = os.path.join(file_path, file_name)
    print("TextEncoder Pre-trained model load from", model_path)
    return model_path


def text_encoder_load_model(model_path, model):
    ckpt = torch.load(model_path)
    state_dict = ckpt['model_state_dict']
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key[0:5] =='bert.':
            new_key = key[5:]
        else:
            new_key = key
        if new_key[0:3] != 'cls':
            new_state_dict[new_key] = value
    ckpt['model_state_dict'] = new_state_dict
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Model fully loaded!")
    return model