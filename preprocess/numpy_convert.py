import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer
import re
from prcs import digit_place_embed
from functools import partial
import tqdm

def pad_list(lst : list):
   return np.pad(lst, (0, 150-len(lst)), 'constant', constant_values=(0))

def numpy_array(lst :list):
    #Convert list to array
    return np.array(lst).tolist()

def word2index(word_list, vocabs):
    return vocabs[word_list]

def re_sub(x):
    return re.sub(r'[,|!?"\':;~()\[\]]', '', x)

def null_fill(df, value_mode):
    def _fillNA(seq, rp_value):
        return [rp_value if x!=x else x for x in seq ]
    
    if value_mode =='VC':
        df['value'] = df['value'].map(lambda x : _fillNA(x, 0.0))
        df['uom'] = df['uom'].map(lambda x : _fillNA(x, ' ')) 
    else: 
        df['value'] = df['value'].map(lambda x : _fillNA(x, ' '))
        df['uom'] = df['uom'].map(lambda x : _fillNA(x, ' '))
    
    return df

def agg_col(df, value_mode):
    def _agg(a, b):
        return [str(x) + str(y) for x,y in zip(a, b)]
    
    def _value_split(x):
    # value seq list
        seq = [' '.join(str(y)) for y in x ]
        return seq 
    
    def _round(seq):
        return [round(x, 6) if type(x)==float else x for x in seq ]
    
    # NV => code_name 
    # VA => code_name + value + uom
    # DSVA => code_name + value(split) + uom
    # VC => code_name + uom / value
    if value_mode == 'NV':
         df['code_name'] = pd.Series([list(map(str, a)) for a in df['code_name']])

    elif value_mode =='VA':
        df['value'] = df['value'].map(lambda x : _round(x))
        df['code_name'] = pd.Series([_agg(a,b) for a, b in zip(df['code_name'], df['value'])])
        df['code_name'] = pd.Series([_agg(a,b) for a, b in zip(df['code_name'], df['uom'])])
   
    elif value_mode =='DSVA':
        df['value'] = df['value'].map(lambda x : _round(x))
        df['value'] = df['value'].map(lambda x : _value_split(x))
        df['code_name'] = pd.Series([_agg(a,b) for a, b in zip(df['code_name'], df['value'])])
        df['code_name'] = pd.Series([_agg(a,b) for a, b in zip(df['code_name'], df['uom'])])

    elif value_mode =='VC':
        df['value'] = df['value'].map(lambda x : _round(x))
        df['code_name'] = pd.Series([_agg(a,b) for a, b in zip(df['code_name'], df['uom'])])

    return df

def making_vocab(df):
    vocab_dict = {}
    vocab_dict['[PAD]'] = 0
    vocab_dict['[CLS]'] = 1
    vocab_dict['[MASK]'] = 2

    df['merge_code_set'] = df['code_name'].apply(lambda x : list(set(x)))
    vocab_set = []
    for codeset in df['merge_code_set']:
        vocab_set.extend(codeset) 
    vocab_set = list(set(vocab_set))
    for idx, vocab in enumerate(vocab_set):
        vocab_dict[vocab] = idx+3
        
    return vocab_dict
                              
def _tokenized_max_length(vocab, tokenizer):
    tokenized_vocab= tokenizer(list(vocab.keys()))
    max_word_len = max(list(map(len, tokenized_vocab['input_ids'])))
    return max_word_len

def _organize(seq):
    return re.sub(r'[,|!?"\':;~()\[\]]', '', seq)

def tokenize_seq(seq, word_max_len, tokenizer):
    seq = list(map(_organize, seq))
    seq = ['[PAD]' if x=='0.0' else x for x in seq]
    tokenized_seq= tokenizer(seq, padding = 'max_length', return_tensors='pt', max_length=word_max_len)
    return tokenized_seq


def convert2numpy(input_path, output_path):
    value_mode_list = ['NV', 'DSVA', 'VC']
    sources = ['mimic','eicu']
    tokenizer= AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    for src in sources:
        save_path = f'{output_path}/input/{src}'
        filename = '{}_df.pkl'.format(src)
        df = pd.read_pickle(os.path.join(input_path, filename))
        print('{} input files load !'.format(src))
        for value_mode in value_mode_list:
            print(value_mode)
            save_name =  f'{src}_input_index_{value_mode}'
            print('save_name', save_name)
            df = null_fill(df, value_mode)
            df = agg_col(df, value_mode)

            vocab = making_vocab(df)
            vocab['0.0'] = 0
            src2index= partial(word2index, vocabs=vocab)
            # input_index 
            index =[list(map(src2index, icu)) for icu in df['code_name']]
            array = np.array(index)
            np.save(os.path.join(save_path, save_name), array)
            
            print('tokenization start!')
            # tokenized
            word_max_len = _tokenized_max_length(vocab, tokenizer)
            token_tmp = [tokenize_seq(seq, word_max_len, tokenizer) for seq in tqdm.tqdm(df['code_name'])]
            df['input_ids'] =pd.Series([token['input_ids'] for token in token_tmp])
            df['token_type_ids'] =pd.Series([token['token_type_ids'] for token in token_tmp])
            df['attention_mask'] =pd.Series([token['attention_mask'] for token in token_tmp])
            
            #tokenized save
            np.save(os.path.join(save_path, f'input_ids_{value_mode}.npy'), np.array(df['input_ids'])) 
            np.save(os.path.join(save_path, f'token_type_ids_{value_mode}.npy'), np.array(df['token_type_ids'])) 
            np.save(os.path.join(save_path, f'attention_mask_{value_mode}.npy'), np.array(df['attention_mask'])) 

            if value_mode == 'NV':
                #value
                value = np.array([df['value']])
                np.save(os.path.join(save_path, 'value.npy'), value[0])


            if value_mode =='DSVA':
                df = digit_place_embed(df, tokenizer)
                np.save(os.path.join(save_path, f'input_ids_DSVA_DPE.npy'), np.array(df['input_ids'])) 
                np.save(os.path.join(save_path, f'token_type_ids_DSVA_DPE.npy'), np.array(df['token_type_ids'])) 
                np.save(os.path.join(save_path, f'attention_mask_DSVA_DPE.npy'), np.array(df['attention_mask']))
        
       
        
        #seq_len
        seq_len = np.array([df['seq_len']])
        np.save(os.path.join(save_path, 'seq_len.npy'), seq_len[0])              

    