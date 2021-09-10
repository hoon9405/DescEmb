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