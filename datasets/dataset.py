import torch
import torch.utils.data

import pandas as pd
import random
import collections
import os
import numpy as np



from transformers import AutoTokenizer

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_type = 'train'):
        self.visualize = args.visualize
        self.type = data_type
        self.pred_model_mode = args.pred_model_mode

        self.textencoder_mask_prob = args.textencoder_mlm_probability
        self.textencoder_ssl = args.textencoder_ssl

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.path = args.input_path
        self.prefix = args.source_file
        if args.concat_type in ['concat_a', 'concat_b', 'concat_c']:
          self.ext = "_" + str(args.concat_type) + ".npy"
        else:
          self.ext = ".npy"
        self.task = args.target
        self.labels = None

        self.seed = args.seed

        if self.type == 'train':
            assert args.ratio != '0', "cannot load train dataset with ratio 0"

        ratio = None if args.ratio in ['0', '100'] else args.ratio
        self.ratio = '' if ratio is None else '_' + ratio

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        raise NotImplementedError()

    def get_fold_indices(self):
        if self.type == 'train':
            hit = 1
        elif self.type == 'valid':
            hit = 2
        else:
            hit = 0
    
        df = pd.read_csv(os.path.join(self.path, "fold", f"{self.prefix}_{self.seed}_fold_split{self.ratio}.csv"))
        print('fold path' , os.path.join(self.path, "fold", f"{self.prefix}_{self.seed}_fold_split{self.ratio}.csv"))
        splits = df[self.task].values
        idcs = np.where(splits == hit)[0]
        if self.visualize ==True:
            return df[self.task].index
        else:
            return idcs

    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: torch.Tensor):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.textencoder_mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.textencoder_mask_prob)

        if special_tokens_mask is None:
            if self.textencoder_ssl == 'reflect_freq':
                special_tokens_mask = [
                        self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                        ]
            else:
                special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
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
    def __init__(self, args, data_type = 'train'):
        super().__init__(
            args = args,
            data_type = data_type
        )
        col_names = ['input_index', 'offset_order', 'seq_len']

        self.value = np.load(
            file = os.path.join(self.path, f"{self.prefix}_input_value.npy")
        )

        self.offset_orders = None
        self.sequential_lengths = None

        self.input_idcs = np.load(
                file = os.path.join(self.path, f"{self.prefix}_input_input_index{self.ext}"),
            )
        if self.prefix.startswith('both_'):
          self.prefix = self.prefix[5:]

        hit_idcs = self.get_fold_indices()

        self.input_idcs = self.input_idcs[hit_idcs]
        self.value = self.value[hit_idcs]

        if self.pred_model_mode == 'transformer':
            self.offset_orders = np.load(
                file = os.path.join(self.path, f"{self.prefix}_input_offset_order.npy"),
            )
            self.offset_orders = self.offset_orders[hit_idcs]
        else:
            self.sequential_lengths = np.load(
                file = os.path.join(args.input_path, f"{self.prefix}_input_seq_len.npy"),
            )
            self.sequential_lengths = self.sequential_lengths[hit_idcs]

        if not self.task.startswith('pretrain'):
            self.labels = np.load(
                file = os.path.join(self.path,'label', f"{self.prefix}_{self.task}_label.npy"),
            )
            print('label path', os.path.join(self.path,'label', f"{self.prefix}_{self.task}_label.npy"))
            self.labels = torch.tensor(self.labels[hit_idcs], dtype=torch.long)
        
    def __len__(self):
        return len(self.input_idcs)
    
    def __getitem__(self, index):
        input_idcs = torch.LongTensor(self.input_idcs[index]) if self.input_idcs is not None else None
        offset_order = torch.LongTensor(self.offset_orders[index]) if self.offset_orders is not None else None
        seq_len = torch.LongTensor(self.sequential_lengths).unsqueeze(-1)[index] if self.sequential_lengths is not None else None
        labels = torch.LongTensor(self.labels).unsqueeze(-1)[index] if self.labels is not None else None
        value = torch.Tensor(self.value[index])    
        
        if offset_order is not None:
            return {
                  'input': input_idcs,
                  'offset_order': offset_order,
                  'value': value,                  
                  'labels': labels,
            } 
        else:
            return {
                  'input': input_idcs,
                  'seq_len': seq_len,
                  'value': value,
                  'labels': labels
            } 

class TokenizedDataset(BaseDataset):
    def __init__(self, args, data_type = 'train'):
        super().__init__(
            args = args,
            data_type = data_type
        )
        col_names = ['input_ids', 'token_type_ids', 'attention_mask']

        # if args.embed_model_mode == 'Scratch-RNN':
        #   col_names[0] +='_rnn'

        self.value = np.load(
            file = os.path.join(self.path, f"{self.prefix}_input_value.npy")
        )

        self.offset_orders = None
        self.sequential_lengths = None

        if self.prefix.startswith('both_'):
          self.prefix = self.prefix[5:]

        if self.textencoder_mask_prob == 0.0:
            hit_idcs = self.get_fold_indices()

            self.input_ids, self.token_type_ids, self.attention_mask = (
                np.load(
                    file = os.path.join(self.path, f"{self.prefix}_tokenized_{col}{self.ext}"),
                ) for col in col_names
            )
            self.input_ids = self.input_ids[hit_idcs]
            self.token_type_ids = self.token_type_ids[hit_idcs]
            self.attention_mask = self.attention_mask[hit_idcs]
            self.value = self.value[hit_idcs]

            if self.pred_model_mode == 'transformer':
                self.offset_orders = np.load(
                    file=os.path.join(self.path, f"{self.prefix}_input_offset_order.npy")
                )
                self.offset_orders = self.offset_orders[hit_idcs]
            else:
                self.sequential_lengths = np.load(
                    file=os.path.join(args.input_path, f"{self.prefix}_input_seq_len.npy"),
                )
                self.sequential_lengths = self.sequential_lengths[hit_idcs]

        else:
            self.input_ids, self.token_type_ids, self.attention_mask = (
                np.load(
                    file = os.path.join(self.path, f"{self.prefix}_tokenized_{col}_uniform.npy"), allow_pickle=True
                ) for col in col_names
            )

        if not self.task.startswith('pretrain'):
            self.labels = np.load(
                file = os.path.join(self.path, 'label', f"{self.prefix}_{self.task}_label.npy"),
            )
            self.labels = torch.tensor(self.labels[hit_idcs], dtype=torch.long)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):

        input_id = torch.LongTensor(self.input_ids[index])
        token_type_id = torch.LongTensor(self.token_type_ids[index])
        attn_mask = torch.LongTensor(self.attention_mask[index])
        offset_order = torch.LongTensor(self.offset_orders[index]) if self.offset_orders is not None else None
        seq_len = (torch.LongTensor(self.sequential_lengths).unsqueeze(-1)[index]) if self.sequential_lengths is not None else None
        labels = torch.LongTensor(self.labels).unsqueeze(-1)[index] if self.labels is not None else None
        value = torch.Tensor(self.value[index])

        if self.type == 'train' and self.textencoder_mask_prob > 0 and offset_order is not None and labels is not None:
            input_id, mlm_labels = self.mask_tokens(input_id, special_tokens_mask=None)

            return {
                'input': input_id,
                'token_type_ids': token_type_id,
                'attention_mask': attn_mask,
                'mlm_labels': mlm_labels,
                'offset_order': offset_order,
                'value': value,
                'labels': labels
            }   
        elif self.type=='train' and self.textencoder_mask_prob > 0 and offset_order is None and labels is not None:
            input_id, mlm_labels = self.mask_tokens(input_id, special_tokens_mask=None)
            return {
                'input': input_id,
                'token_type_ids': token_type_id,
                'attention_mask': attn_mask,
                'mlm_labels': mlm_labels,
                'seq_len': seq_len,
                'value': value,
                'labels': labels
            }

        elif (self.textencoder_mask_prob == 0 or self.type!='train') and offset_order is not None and labels is not None:
            return {
                'input': input_id,
                'token_type_ids': token_type_id,
                'attention_mask': attn_mask,
                'offset_order': offset_order,
                'value': value,
                'labels': labels
            }
        elif labels is None:
            input_id, mlm_labels = self.mask_tokens(input_id, special_tokens_mask=None)
            return {
                'input': input_id,
                'token_type_ids': token_type_id,
                'attention_mask': attn_mask,
                'mlm_labels': mlm_labels
            }

        else:
            return {
                'input': input_id,
                'token_type_ids': token_type_id,
                'attention_mask': attn_mask,
                'seq_len': seq_len,
                'value': value,
                'labels': labels
            }


class SSLDataset(BaseDataset):
    def __init__(self, args, data_type):
        super().__init__(
            args = args,
            data_type = data_type
        )
        
        col_names = ['input_index', 'offset_order', 'seq_len']

        self.input_idcs, self.offset_orders, self.sequential_lengths = (
            np.load(
                file = os.path.join(self.path, f"{self.prefix}_input_SSL_{col}.npy"),
            ) for col in col_names
        )

    def __len__(self):
        return len(self.input_idcs)
    
    def __getitem__(self, index):
        return {
            'input': torch.LongTensor(self.input_idcs[index]),
            'offset_order': torch.LongTensor(self.offset_orders[index]),
            'seq_len': torch.LongTensor([self.sequential_lengths[index]]),
        }

class TokenizedSSLDataset(BaseDataset):
    def __init__(self, args, data_type):
        super().__init__(
            args = args,
            data_type = data_type
        )

        col_names = ['input_ids', 'token_type_ids', 'attention_mask']

        self.offset_orders = None
        self.sequential_lengths = None
        if self.textencoder_ssl == 'reflect_freq':
            self.input_ids, self.token_type_ids, self.attention_mask = (
                np.load(
                    file = os.path.join(self.path, f"{self.prefix}_tokenized_SSL_{col}.npy"),
                ) for col in col_names
            )
            if args.pred_model_mode:
                self.offset_orders = np.load(
                    file = os.path.join(self.path, f"{self.prefix}_input_SSL_offset_order.npy"),
                )
            else:
                self.sequential_lengths = np.load(
                    file = os.path.join(args.input_path, f"{self.prefix}_input_SSL_seq_len.npy"),
                )
        else:
            self.input_ids, self.token_type_ids, self.attention_mask = (
                np.load(
                    file = os.path.join(self.path, f"{self.prefix}_tokenized_SSL_unique_{col}.npy"),
                ) for col in col_names
            )

        self.sequential_lengths = np.load(
            file = os.path.join(args.input_path, f"{self.prefix}_input_SSL_seq_len.npy"),
        )

        #XXX DEBUG!
        # tentatively applied
        
        idcs = np.where(self.sequential_lengths == 150)
        self.offset_orders = self.offset_orders[idcs]
        self.input_ids = self.input_ids[idcs]
        self.token_type_ids = self.token_type_ids[idcs]
        self.attention_mask = self.attention_mask[idcs]
      

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        input_id = torch.LongTensor(self.input_ids[index])
        token_type_id = torch.LongTensor(self.token_type_ids[index])
        attn_mask = torch.LongTensor(self.attention_mask[index])
        offset_order = torch.LongTensor(self.offset_orders[index]) if self.offset_orders is not None else None
        seq_len = torch.LongTensor(self.sequential_lengths.unsqueeze(-1)[index]) if self.sequential_lengths is not None else None

        if self.textencoder_mask_prob > 0:
            input_id, mlm_labels = self.mask_tokens(input_id, special_tokens_mask = None)
            return {
                'input': input_id,
                'token_type_ids': token_type_id,
                'attention_mask': attn_mask,
                'mlm_labels': mlm_labels,
            }
        elif offset_order is not None:
            return {
                'input': input_id,
                'token_type_ids': token_type_id,
                'attention_mask': attn_mask,
                'offset_order': offset_order,
            }
        else:
            return {
                'input_ids': input_id,
                'token_type_ids': token_type_id,
                'attention_mask': attn_mask,
                'seq_len': seq_len,
            }


class Word2VecDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        data = args.dataset
        self.task = args.task
        # TODO 변경하기
        self.path = args.input_path
        if data == 'mimic':
            file_path = os.path.join(self.path, 'mimic_input_input_index')
            file_path = file_path + f'_{args.value_embed_type}.npy'
        elif data == 'eicu':
            file_path = os.path.join(self.path, f'eicu_input_input_index')
            file_path = file_path + f'_{args.value_embed_type}.npy'

        data = np.load(file_path)
        data = self.indexing(data, args.dataset, args.seed)
        self.pos_pair, self.neg_pair = self.preprocess(data)
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