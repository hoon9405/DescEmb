import os
import logging
import pprint
import wandb
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
        # self.args = args
        self.device = args.device
        self.input_path = args.input_path
        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix
        
        self.data = args.data
        self.eval_data = args.eval_data
        self.value_embed_type = args.value_embed_type
        self.valid_subsets = args.valid_subsets
        self.fold = args.fold
        self.task = args.task

        self.model_type = args.model

        # self.debug = args.debug
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

        # if not self.debug:
        #     wandb.init(project=args.wandb_project_name, entity="pretrained_ehr", config=args, reinit=True)

        self.best_eval_path = args.save_path + '_best_auprc.pt'
        self.best_mimic_eval_path = args.save_path + '_mimic_best_auprc.pt'
        self.best_eicu_eval_path = args.save_path + '_eicu_best_auprc.pt'

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

        #Model define
        #XXX
        if args.transfer:
            if os.path.exists(args.model_path):
                logger.info(
                    f"transferring pretrained model from {args.model_path}"
                )
                state_dict = torch.load(args.model_path)['model_state_dict']
                #XXX breakpoint()
                if args.embed_model_mode in ['CodeEmb', 'Bert-FT']:
                    state_dict = {k.replace('module.',''): state_dict[k] for k in state_dict if (not 'embedding' in k and not 'bert_embed' in k)}
                else:
                    state_dict = {k.replace('module.',''): state_dict[k] for k in state_dict }

                model.load_state_dict(state_dict, strict = False)
            else:
                raise FileNotFoundError(
                    f"--model_path {args.model_path} does not exist"
                )
        self.model = nn.DataParallel(model).to(self.device)
       
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
                ratio=self.ratio
            )
        #XXX word2vec dataset
        # elif self.task == 'w2v':
        #     dataset = Word2VecDataset()
        #     ...
        elif self.model_type.startswith('codeemb'):
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
        elif self.model_type.startswith('descemb'):
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
            raise NotImplementedError(), self.model_type

        self.data_loaders[split] = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def train(self):
        for epoch in range(1, self.n_epochs + 1):
            logger.info(f"begin training epoch {epoch}")
            preds_train = []
            truths_train = []
            total_train_loss = 0

            self.model.train()

            for sample in tqdm.tqdm(self.train_dataloader):
                self.optimizer.zero_grad(set_to_none=True)

                net_output = self.model(**sample["net_input"])
                logits = self.model.get_logits(net_output)
                target = self.get_targets(sample)

                #XXX breakpoint() -> label.shape
                #XXX check when mlm
                if self.target == 'diagnosis':
                    loss = self.criterion(logits, target.squeeze(2))
                else:
                    loss = self.criterion(logits, target)

                #XXX mlm -> task
                #XXX mlm + prediction task simultaneously? -> x
                # if self.mlm_prob > 0:
                #     mlm_labels = sample['mlm_labels'].to(self.device)
                #     mlm_labels = mlm_labels.view(-1)
                #     mlm_output = mlm_output.view(-1, 28996)

                #     survivor = torch.where(mlm_labels != -100)[0]

                #     mlm_labels = mlm_labels[survivor]
                #     mlm_output = mlm_output[survivor]

                #     extra_loss = F.cross_entropy(mlm_output, mlm_labels)
                #     loss += extra_loss

                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

                probs_train = torch.sigmoid(logits).detach().cpu().numpy()
                preds_train += list(probs_train.flatten())
                truths_train += list(target.detach().cpu().numpy().flatten())         

            avg_train_loss = total_train_loss / len(self.train_dataloader)
            auroc_train = roc_auc_score(truths_train, preds_train)
            auprc_train = average_precision_score(truths_train, preds_train, average='micro')

            # if not self.debug:
            #     wandb.log(
            #         {
            #             'train_loss': avg_train_loss,
            #             'train_auroc': auroc_train,
            #             'train_auprc': auprc_train
            #         }
            #     )

            #XXX
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

        valid_auprcs = []
        for subset in valid_subsets:
            logger.info("begin validation on '{}' subset".format(subset))

            for sample in self.data_loaders[subset]:
                with torch.no_grad():
                    net_output = self.model(**sample["net_input"])
                    logits = self.model.get_logits(net_output)
                    target = self.get_targets(sample)

                    #XXX breakpoint() -> label.shape
                    #XXX check when mlm
                    if self.target == 'diagnosis':
                        loss = self.criterion(logits, target.squeeze(2))
                    else:
                        loss = self.criterion(logits, target)

                total_valid_loss += loss.item()

                probs_valid = torch.sigmoid(logits).detach().cpu().numpy()
                preds_valid += list(probs_valid.flatten())
                truths_valid += list(target.detach().cpu().numpy().flatten())

            avg_valid_loss = total_valid_loss / len(self.data_loaders[subset])
            auroc_valid = roc_auc_score(truths_valid, preds_valid)
            auprc_valid = average_precision_score(truths_valid, preds_valid, average='micro')

            with rename_logger(logger, subset):
                logger.info(
                    "epoch: {}, loss: {:.3f}, auroc: {:.3f}, auprc: {:.3f}".format(
                        epoch, avg_valid_loss, auroc_valid, auprc_valid
                    )
                )
            # if not self.debug:
            #     wandb.log({'eval_loss': avg_valid_loss,
            #                 'eval_auroc': auroc_valid,
            #                 'eval_auprc': auprc_valid})

            valid_auprcs.append(auprc_valid)

        return valid_auprcs

    def validate_and_save(
        self,
        epoch,
        valid_subsets
    ):
        should_stop = False

        #TODO add more options for validation (e.g. validate_metric, validate_interval, ...)
        valid_auprcs = self.validate(epoch, valid_subsets)
        should_stop |= should_stop_early(self.args.patience, valid_auprcs[0])

        prev_best = getattr(should_stop_early, "best", None)
        if prev_best and prev_best == valid_auprcs[0]:
            logger.info(
                "Saving checkpoint to {}".format(
                    os.path.join(self.args.save_dir, self.args.save_prefix + "_best.pt")
                )
            )
            torch.save(
                {
                    'model_state_dict': self.model.module.state_dict() if (
                        isinstance(self.model, DataParallel)
                    ) else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epochs': epoch,
                    # 'loss': best_loss,
                    # 'auroc': best_auroc,
                    # 'auprc': best_auprc,
                },
                os.path.join(self.args.save_dir, self.args.save_prefix + "_best.pt")
            )
            logger.info(
                "Finished saving checkpoint to {}".format(
                    os.path.join(self.args.save_dir, self.args.save_prefix + "_best.pt")
                )
            )

        return should_stop