import os
import logging
import pprint
import tqdm

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader

from utils.trainer_utils import (
    rename_logger,
    should_stop_early
)
from datasets import (
  Dataset,
  TokenizedDataset,
  MLMTokenizedDataset
)
import models

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.input_path = args.input_path
        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix

        self.disable_validation = args.disable_validation
        self.patience = args.patience

        self.data = args.data
        self.eval_data = args.eval_data
        self.value_embed_type = args.value_embed_type
        self.valid_subsets = args.valid_subsets
        self.fold = args.fold
        self.task = args.task

        self.model_type = args.model
        self.embed_model_type = args.embed_model

        self.seed = args.seed
        self.ratio = args.ratio
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs

        self.mlm_prob = args.mlm_prob if self.task == 'mlm' else 0

        self.rnn_layer = args.rnn_layer
        self.dropout = args.dropout
        self.pred_embed_dim = args.pred_embed_dim
        self.pred_hidden_dim = args.pred_hidden_dim

        self.data_loaders = dict()

        # print args
        logger.info(pprint.pformat(vars(args)))

        model = models.build_model(args)

        logger.info(model)
        logger.info("task: {}".format(self.task))
        logger.info("model: {}".format(model.__class__.__name__))
        logger.info(
            "num. model params: {:,} (num. trained: {:,})".format(
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            )
        )

        self.model = nn.DataParallel(model, device_ids=args.device_ids).to('cuda')

        for subset in ['train'] + self.valid_subsets:
            self.load_dataset(subset)

        self.criterion = (
            nn.BCEWithLogitsLoss() if (
                self.task not in ["mlm", "w2v"]
            ) else nn.CrossEntropyLoss()
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def load_dataset(self, split: str):
        if self.task == 'mlm':
            dataset = MLMTokenizedDataset(
                input_path=self.input_path,
                data=self.data,
                eval_data=self.eval_data,
                fold=self.fold,
                split=split,
                value_embed_type=self.value_embed_type,
                task=self.task,
                seed=self.seed,
                ratio=self.ratio,
                mlm_prob=self.mlm_prob
            )
        elif self.embed_model_type.startswith('codeemb'):
            dataset = Dataset(
                input_path=self.input_path,
                data=self.data,
                eval_data=self.eval_data,
                fold=self.fold,
                split=split,
                value_embed_type=self.value_embed_type,
                task=self.task,
                seed=self.seed,
                ratio=self.ratio
            )
        elif self.embed_model_type.startswith('descemb'):
            dataset = TokenizedDataset(
                input_path=self.input_path,
                data=self.data,
                eval_data=self.eval_data,
                fold=self.fold,
                split=split,
                value_embed_type=self.value_embed_type,
                task=self.task,
                seed=self.seed,
                ratio=self.ratio
            )
        else:
            raise NotImplementedError(self.model_type)

        self.data_loaders[split] = DataLoader(
            dataset, collate_fn=dataset.collator, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def train(self):
        for epoch in range(1, self.n_epochs + 1):
            logger.info(f"begin training epoch {epoch}")
            preds_train = []
            truths_train = []
            total_train_loss = 0
            auroc_train = 0
            auprc_train = 0

            self.model.train()

            for sample in tqdm.tqdm(self.data_loaders['train']):
                self.optimizer.zero_grad(set_to_none=True)

                net_output = self.model(**sample["net_input"])
                #NOTE we assume self.model is wrapped by torch.nn.parallel.data_parallel.DataParallel
                logits = self.model.module.get_logits(net_output)
                target = self.model.module.get_targets(sample).to(logits.device)

                if self.task == 'diagnosis':
                    loss = self.criterion(logits, target.squeeze(2))
                else:
                    loss = self.criterion(logits, target)

                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

                with torch.no_grad():
                    if self.task not in ['mlm', 'w2v']:
                        truths_train += list(target.cpu().numpy().flatten())
                        probs_train = torch.sigmoid(logits).cpu().numpy()
                        preds_train += list(probs_train.flatten())

            avg_train_loss = total_train_loss / len(self.data_loaders['train'])
            if self.task not in ['mlm', 'w2v']:
                auroc_train = roc_auc_score(truths_train, preds_train)
                auprc_train = average_precision_score(truths_train, preds_train, average='micro')

            with rename_logger(logger, "train"):
                logger.info(
                    "epoch: {}, loss: {:.3f}, auroc: {:.3f}, auprc: {:.3f}".format(
                        epoch, avg_train_loss, auroc_train, auprc_train
                    )
                )

            should_stop = self.validate_and_save(epoch, self.valid_subsets)
            if should_stop:
                break
        
    def validate(
        self,
        epoch,
        valid_subsets
    ):
        self.model.eval()

        preds_valid = []
        truths_valid = []
        total_valid_loss = 0
        auroc_valid = 0
        auprc_valid = 0

        valid_auprcs = []
        for subset in valid_subsets:
            logger.info("begin validation on '{}' subset".format(subset))

            for sample in self.data_loaders[subset]:
                with torch.no_grad():
                    net_output = self.model(**sample["net_input"])
                    #NOTE we assume self.model is wrapped by torch.nn.parallel.data_parallel.DataParallel
                    logits = self.model.module.get_logits(net_output)
                    target = self.model.module.get_targets(sample).to(logits.device)

                    if self.task == 'diagnosis':
                        loss = self.criterion(logits, target.squeeze(2))
                    else:
                        loss = self.criterion(logits, target)

                total_valid_loss += loss.item()

                with torch.no_grad():
                    if self.task not in ['mlm', 'w2v']:
                        truths_valid += list(target.cpu().numpy().flatten())
                        probs_valid = torch.sigmoid(logits).cpu().numpy()
                        preds_valid += list(probs_valid.flatten())

            avg_valid_loss = total_valid_loss / len(self.data_loaders[subset])
            if self.task not in ['mlm', 'w2v']:
                auroc_valid = roc_auc_score(truths_valid, preds_valid)
                auprc_valid = average_precision_score(truths_valid, preds_valid, average='micro')

            with rename_logger(logger, subset):
                logger.info(
                    "epoch: {}, loss: {:.3f}, auroc: {:.3f}, auprc: {:.3f}".format(
                        epoch, avg_valid_loss, auroc_valid, auprc_valid
                    )
                )

            valid_auprcs.append(auprc_valid)

        return valid_auprcs

    def validate_and_save(
        self,
        epoch,
        valid_subsets
    ):
        if (
            self.disable_validation
            or valid_subsets is None
            or self.task in ['mlm', 'w2v']
        ):
            logger.info(
                "Saving checkpoint to {}".format(
                    os.path.join(self.save_dir, self.save_prefix + "_last.pt")
                )
            )
            torch.save(
                {
                    'model_state_dict': self.model.module.state_dict() if (
                        isinstance(self.model, DataParallel)
                    ) else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epochs': epoch,
                    'args': self.args
                },
                os.path.join(self.save_dir, self.save_prefix + "_last.pt")
            )
            logger.info(
                "Finished saving checkpoint to {}".format(
                    os.path.join(self.save_dir, self.save_prefix + "_last.pt")
                )
            )
            return False

        should_stop = False

        #TODO add more options for validation (e.g. validate_metric, validate_interval, ...)
        valid_auprcs = self.validate(epoch, valid_subsets)
        should_stop |= should_stop_early(self.patience, valid_auprcs[0])

        prev_best = getattr(should_stop_early, "best", None)
        if (
            self.patience <= 0
            or prev_best is None
            or (prev_best and prev_best == valid_auprcs[0])
        ):
            logger.info(
                "Saving checkpoint to {}".format(
                    os.path.join(self.save_dir, self.save_prefix + "_best.pt")
                )
            )
            torch.save(
                {
                    'model_state_dict': self.model.module.state_dict() if (
                        isinstance(self.model, DataParallel)
                    ) else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epochs': epoch,
                    'args': self.args,
                },
                os.path.join(self.save_dir, self.save_prefix + "_best.pt")
            )
            logger.info(
                "Finished saving checkpoint to {}".format(
                    os.path.join(self.save_dir, self.save_prefix + "_best.pt")
                )
            )

        return should_stop