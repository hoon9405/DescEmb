import os
import logging

import numpy as np
import torch
from collections import OrderedDict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def should_stop_early(patience, valid_auprc: float) -> bool:
    if valid_auprc is None:
        return False
    if patience <= 0:
        return False

    # add flexibility for applying various metrics in the future (e.g. loss, ...)
    def is_better(a, b):
        return a > b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_auprc, prev_best):
        should_stop_early.best = valid_auprc
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    patience
                )
            )
            return True
        else:
            return False

@contextmanager
def rename_logger(logger, new_name):
    old_name = logger.name
    if new_name is not None:
        logger.name = new_name
    yield logger
    logger.name = old_name

# legacy
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