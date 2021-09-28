import os
import logging
import random
import collections

import torch
import torch.utils.data

import numpy as np
import pandas as pd

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_path,
        data,
        eval_data,
        fold,
        split,
        value_embed_type,
        task,
        seed,
        ratio,
    ):
        assert (
            task not in ["mlm", "w2v"]
            or not (data == 'pooled' and eval_data)
        ), "--eval_data should be set if pooled-learning on prediction tasks"

        self.data = data
        if task in ["mlm", "w2v"]:
            eval_data = data

        self.input_path = input_path
        self.split = split
        self.prefix = (
            eval_data if (
                data == 'pooled' and split != 'train'
            ) else data
        )
        self.data_path = os.path.join(self.input_path, self.prefix)
        self.label_path = os.path.join(self.input_path, "label")


        self.ext = "_" + str(value_embed_type) + ".npy"
        self.task = task
        self.seed = seed


        self.labels = None
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.ratio = '' if ratio == '100' else '_' + ratio

        if fold:
            self.fold = fold
        else:
            self.fold = os.path.join(
                self.input_path, "fold", "{}_{}_fold_split{}.csv".format(
                    self.prefix, self.seed, self.ratio
                )
            )

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        raise NotImplementedError()

    def get_fold_indices(self):
        if self.split == 'train':
            hit = 1
        elif self.split == 'valid':
            hit = 2
        elif self.split == 'test':
            hit = 0
    
        df = pd.read_csv(self.fold)
        splits = df[self.task].values
        idcs = np.where(splits == hit)[0]
        return idcs

    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: torch.Tensor):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.textencoder_mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)

        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(
                    labels, already_has_special_tokens=True
                ),
                dtype=torch.bool
            )
        else:
            special_tokens_mask = special_tokens_mask.bool()
       
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        while torch.equal(masked_indices, torch.zeros(len(masked_indices)).bool()):
            masked_indices = torch.bernoulli(probability_matrix).bool()
    
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class Dataset(BaseDataset):
    def __init__(
        self,
        input_path,
        data,
        eval_data,
        fold,
        split,
        value_embed_type,
        task,
        seed,
        ratio,
    ):
        super().__init__(
            input_path=input_path,
            data=data,
            eval_data=eval_data,
            fold=fold,
            split=split,
            value_embed_type=value_embed_type,
            task=task,
            seed=seed,
            ratio=ratio,
        )
        hit_idcs = self.get_fold_indices()

        self.sequential_lengths = None

        self.value = np.load(
            file=os.path.join(self.data_path, "value.npy")
        )
        self.value = self.value[hit_idcs]

        if self.data == 'pooled':
            self.input_idcs = np.load(
                file=os.path.join(self.data_path, "pooled_input_index{}".format(self.ext)),
            )
        else:
            self.input_idcs = np.load(
                file=os.path.join(self.data_path, "{}_input_index{}".format(self.prefix, self.ext))
            )
        self.input_idcs = self.input_idcs[hit_idcs]

        self.sequential_lengths = np.load(
            file=os.path.join(self.data_path, f"seq_len.npy"),
        )
        self.sequential_lengths = self.sequential_lengths[hit_idcs]

        self.label = np.load(
            file=os.path.join(
                self.label_path, "{}_{}_label.npy".format(self.prefix, self.task)
            ).format(self.prefix, self.task),
        )
        self.label = torch.tensor(self.label[hit_idcs], dtype=torch.long)
        
        logger.info(f"loaded {len(self.input_idcs)} {self.split} samples")

    def __len__(self):
        return len(self.input_idcs)
    
    def __getitem__(self, index):
        input_idcs = torch.LongTensor(self.input_idcs[index])
        seq_len = torch.LongTensor(self.sequential_lengths).unsqueeze(-1)[index]
        label = torch.LongTensor(self.label).unsqueeze(-1)[index]
        value = torch.Tensor(self.value[index])    
        
        return {
                'input_ids': input_idcs,
                'seq_len': seq_len,
                'value': value,
                'label': label
        }
    
    def collator(self, samples):
        samples = [s for s in samples if s["input_ids"] is not None]
        if len(samples) == 0:
            return {}
        
        input = {"input_ids": torch.stack([s["input_ids"] for s in samples])}
        input["seq_len"] = torch.stack([s["seq_len"] for s in samples])
        input["value"] = torch.stack([s["label"] for s in samples])
        out = {"label": torch.stack([s["label"] for s in samples])}

        out["net_input"] = input
        return out

class TokenizedDataset(BaseDataset):
    def __init__(
        self,
        input_path,
        data,
        eval_data,
        fold,
        split,
        value_embed_type,
        task,
        seed,
        ratio,
    ):
        super().__init__(
            input_path=input_path,
            data=data,
            eval_data=eval_data,
            fold=fold,
            split=split,
            value_embed_type=value_embed_type,
            task=task,
            seed=seed,
            ratio=ratio,
        )

        hit_idcs = self.get_fold_indices()
        col_names = ['input_ids', 'token_type_ids', 'attention_mask']

        self.offset_orders = None
        self.sequential_lengths = None

        self.value = np.load(
            file=os.path.join(self.data_path, "value.npy")
        )
        self.value = self.value[hit_idcs]

        self.input_ids, self.token_type_ids, self.attention_mask = (
            np.load(
                file=os.path.join(self.data_path, f"{col}{self.ext}"),
            ) for col in col_names
        )
        self.input_ids = self.input_ids[hit_idcs]
        self.token_type_ids = self.token_type_ids[hit_idcs]
        self.attention_mask = self.attention_mask[hit_idcs]

        self.sequential_lengths = np.load(
            file=os.path.join(self.data_path, "seq_len.npy"),
        )
        self.sequential_lengths = self.sequential_lengths[hit_idcs]


        self.label = np.load(
            file=os.path.join(
                self.label_path, "{}_{}_label.npy".format(self.prefix, self.task)
            ).format(self.prefix, self.task),
        )
        self.label = torch.tensor(self.label[hit_idcs], dtype=torch.long)
    
        logger.info(f"loaded {len(self.input_ids)} {self.split} samples")

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        input_ids = torch.LongTensor(self.input_ids[index])
        token_type_id = torch.LongTensor(self.token_type_ids[index])
        attn_mask = torch.LongTensor(self.attention_mask[index])
        seq_len = (torch.LongTensor(self.sequential_lengths).unsqueeze(-1)[index])
        label = torch.LongTensor(self.label).unsqueeze(-1)[index]
        value = torch.Tensor(self.value[index])

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_id,
            'attention_mask': attn_mask,
            'seq_len': seq_len,
            'value': value,
            'label': label
        }
    
    def collator(self, samples):
        samples = [s for s in samples if s["input_ids"] is not None]
        if len(samples) == 0:
            return {}

        input = {"input_ids": torch.stack([s["input_ids"] for s in samples])}
        input["token_type_ids"] = torch.stack([s["token_type_ids"] for s in samples])
        input["attention_mask"] = torch.stack([s["attention_mask"] for s in samples])
        input["seq_len"] = torch.stack([s["seq_len"] for s in samples])
        input["value"] = torch.stack([s["value"] for s in samples])
        out = {"label": torch.stack([s["label"] for s in samples])}

        out["net_input"] = input
        return out

class MLMTokenizedDataset(BaseDataset):
    def __init__(
        self,
        input_path,
        data,
        eval_data,
        fold,
        split,
        value_embed_type,
        task,
        seed,
        ratio,
        mlm_prob
    ):
        super().__init__(
            input_path=input_path,
            data=data,
            eval_data=eval_data,
            fold=fold,
            split=split,
            value_embed_type=value_embed_type,
            task=task,
            seed=seed,
            ratio=ratio,
        )
        self.mlm_prob = mlm_prob

        col_names = ['input_ids', 'token_type_ids', 'attention_mask']

        self.input_ids, self.token_type_ids, self.attention_mask = (
            np.load(
                file=os.path.join(self.data_path, f"{col}_unique_code.npy")
            ) for col in col_names
        )
        logger.info(f"loaded {len(self.input_ids)} {self.split} samples")

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        input_ids = torch.LongTensor(self.input_ids[index])
        token_type_id = torch.LongTensor(self.token_type_ids[index])
        attn_mask = torch.LongTensor(self.attention_mask[index])

        input_ids, mlm_labels = self.mask_tokens(input_ids, special_tokens_mask=None)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_id,
            'attention_mask': attn_mask,
            'mlm_labels': mlm_labels,
        }
    
    def collator(self, samples):
        samples = [s for s in samples if s["input_ids"] is not None]
        if len(samples) == 0:
            return {}
        
        input = {"input_ids": torch.stack([s["input_ids"] for s in samples])}
        input["token_type_ids"] = torch.stack([s["token_type_ids"] for s in samples])
        input["attention_mask"] = torch.stack([s["attention_mask"] for s in samples])
        out = {"label": torch.stack([s["mlm_labels"] for s in samples])}

        out["net_input"] = input
        return out

class Word2VecDataset(BaseDataset):
    def __init__(
        self,
        input_path,
        data,
        eval_data,
        fold,
        split,
        value_embed_type,
        task,
        seed,
        ratio,
    ):
        super().__init__(
            self,
            input_path,
            data,
            eval_data,
            fold,
            split,
            value_embed_type,
            task,
            seed,
            ratio,
        )

        input_idcs = np.load(
            file=os.path.join(self.data_path, "{}_input_index{}".format(self.prefix, self.ext))
        )
        input_idcs = self.indexing(input_idcs, self.data, self.seed)
        self.pos_pair, self.neg_pair = self.preprocess(input_idcs)
        self.pos_pair.pop(0)
        self.index_dict = {i: k for i, k in enumerate(self.pos_pair.keys())}

    def __len__(self):
        return len(self.pos_pair)

    def __getitem__(self, item):
        item = self.index_dict[item]
        try:
            pos = random.sample(self.pos_pair[item], 5)
        except ValueError:
            pos = random.choices(self.pos_pair[item], k=5)

        try:
            neg = random.sample(self.neg_pair[item], 30)
        except ValueError:
            neg = random.choices(self.neg_pair[item], k=30)

        return torch.LongTensor([item]), torch.LongTensor(pos), torch.LongTensor(neg)

    def indexing(self, data, dataname, seed):
        hit = 1

        df = pd.read_csv(os.path.join(self.path, 'fold', f'{dataname}_{seed}_fold_split.csv'))
        splits = df[self.task].values
        idcs = np.where(splits == hit)[0]

        data = data[idcs]
        return data

    def preprocess(self, mimic):
        pos_pair = {}
        skip_window=15
        for index in range(mimic.shape[0]):
            data = mimic[index]
            data_index = 0
            span = 2 * skip_window + 1  # [ skip_window, target, skip_window ]
            buffer = collections.deque(maxlen=span)
            for _ in range(span):
                buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data)

            for m in range(skip_window):
                try:
                    pos_pair[buffer[m]].extend(list(set(list(buffer))))
                except KeyError:
                    pos_pair[buffer[m]] = list(set(list(buffer)))

            for i in range(np.nonzero(data)[0].max() - skip_window):
                key = buffer[skip_window]
                if buffer[skip_window] == 0:
                    continue
                try:
                    pos_pair[key].extend(list(set(list(buffer))))
                except KeyError:
                    pos_pair[key] = list(set(list(buffer)))

                buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data)
            data_index = (data_index + len(data) - span) % len(data)

        pos_pair = {k: list(set(v)) for k, v in pos_pair.items()}

        # negative_pair
        max_num = mimic.max()

        neg_pair = {k:list(set(v) ^ set(list(np.arange(3, max_num)))) for k, v in pos_pair.items()}

        return pos_pair, neg_pair