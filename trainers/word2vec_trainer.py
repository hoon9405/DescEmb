import os
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.dataset import Word2VecDataset
from models.word2vec import Word2VecModel
from utils.trainer_utils import rename_logger, EarlyStopping
import pickle

logger = logging.getLogger(__name__)

class Word2VecTrainer():
    def __init__(self, args):

        self.input_path = args.input_path
        self.data = args.src_data
        self.eval_data = args.eval_data
        self.value_mode = args.value_mode
        self.valid_subsets = args.valid_subsets
        self.fold = args.fold
        self.task = args.task

        self.seed = args.seed
        self.ratio = args.ratio
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs

        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix

        if args.src_data == 'pooled':
            mimic_dict = self.vocab_load(args.input_path, 'mimiciii', args.value_mode)
            eicu_dict = self.vocab_load(args.input_path, 'eicu', args.value_mode)
            index_size = len(mimic_dict) + len(eicu_dict) - 3
        else:
            vocab_dict = self.vocab_load(args.input_path, args.src_data, args.value_mode)
            index_size = len(vocab_dict)

        dataset = Word2VecDataset(
            input_path=self.input_path,
            data=self.data,
            eval_data=self.eval_data,
            fold=self.fold,
            split=self.valid_subsets,
            value_mode=self.value_mode,
            task=self.task,
            seed=self.seed,
            ratio=self.ratio
        )

        self.dataloader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True
            )

        self.model = Word2VecModel(index_size, emb_dim=args.enc_embed_dim).cuda()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

        self.early_stopping = EarlyStopping(patience=20, verbose=True)

    def train(self):
        best_loss = float('inf')
        for epoch in range(self.n_epochs):
            avg_loss = 0
            for iter, sample in enumerate(self.dataloader):
                batch_input, batch_labels, batch_neg = sample
                batch_input = batch_input.cuda()
                batch_labels = batch_labels.cuda()
                batch_neg = batch_neg.cuda()
                
                loss = self.model(batch_input, batch_labels, batch_neg)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()

            avg_loss /= len(self.dataloader)

            logger.info(
                "epoch: {}, loss: {:.3f}".format(
                    epoch, avg_loss
                )
            )

            self.early_stopping(-avg_loss)
            if best_loss > avg_loss:
                best_loss = avg_loss
                logger.info(
                    "Saving checkpoint to {}".format(
                        os.path.join(self.save_dir, self.save_prefix + "_best.pt")
                    )
                )
                torch.save(
                    {
                        'model_state_dict': self.model.state_dict()
                    },
                    os.path.join(self.save_dir, self.save_prefix + "_best.pt")
                )
                logger.info(
                    "Finished saving checkpoint to {}".format(
                        os.path.join(self.save_dir, self.save_prefix + "_best.pt")
                    )
                )

            if self.early_stopping.early_stop is True:
                break

    def vocab_load(self, data_path, src_data, value_mode):
        vocab_path = os.path.join(
            data_path, src_data, f'code_index_{value_mode}_vocab.pkl'
            )
        with open(vocab_path, 'rb') as file:
            vocab_dict = pickle.load(file)
        return vocab_dict
    