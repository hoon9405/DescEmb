import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer
import re
from prcs import digit_place_embed
from functools import partial
import tqdm
import pickle
import torch

def numpy_array(lst :list):
    #Convert list to array
    return np.array(lst).tolist()

def word2index(word_list, vocabs):
    return vocabs[word_list]

def re_sub(x):
    return re.sub(r'[,|!?"\':;~()\[\]]', '', x)

def null_fill(df, value_mode):
    def _fillNA(seq, rp_value):
        return [rp_value if not x or x==' ' else x for x in seq ]
    
    # Value Concat(VC) mode only fill NA with 0.0
    # Others fill NA with empty space
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
    
    def _convert_to_float(mixed_list):
        converted_list = []
        
        for item in mixed_list:
            if isinstance(item, str):
                try:
                    converted_list.append(float(item))
                except ValueError:
                    converted_list.append(0.0)
            elif isinstance(item, float) or isinstance(item, int):
                converted_list.append(item)
            else:
                converted_list.append(0.0)
        
        return converted_list

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
        df['value'] = df['value'].map(lambda x : _convert_to_float(x))
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


def convert2numpy(dest_path, src_data, value_mode, data_type):
    tokenizer= AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    filename = f'{src_data}_df.pkl'
    df = pd.read_pickle(os.path.join(dest_path, filename))
    print(f'{src_data} dataframe pickle file has been loaded')
    print('value mode : ', value_mode)
    
    save_path = os.path.join(dest_path, src_data)
    os.makedirs(save_path, exist_ok=True)
    
    codeemb_feature_save_name =  f'code_index_{value_mode}'
    print('codeemb feature index save_name', codeemb_feature_save_name)
    
    df = null_fill(df, value_mode)
    # diffentiate code_name and value aggregation for each value mode
    df = agg_col(df, value_mode) 

    # indexing code features for preparing codeemb input
    vocab = making_vocab(df)
    vocab['0.0'] = 0
    
    vocab_save_path = os.path.join(save_path, f'{codeemb_feature_save_name}_vocab.pkl')
    with open(vocab_save_path, 'wb') as file:
        pickle.dump(vocab, file)
    
    #Event seqeunce_length save 
    seq_len = np.array([df['seq_len']])
    np.save(os.path.join(save_path, 'seq_len.npy'), seq_len[0])    
       
    src2index= partial(word2index, vocabs=vocab)
    # CodeEmb feature save
    
    index =[list(map(src2index, icu)) for icu in df['code_name']]
    array = np.array(index)         
    np.save(os.path.join(save_path, codeemb_feature_save_name), array)

    if data_type == 'pretrain':
        keys_to_remove = ['[PAD]', '[MASK]', '[CLS]']
        for key in keys_to_remove:
            vocab.pop(key, None) 
        df= pd.DataFrame({'code_name': [list(vocab.keys())]})
        
    print(f'tokenization for preparing descemb input with {value_mode} mode')
    # tokenization for preparing descemb input
    word_max_len = _tokenized_max_length(vocab, tokenizer)
    token_tmp = [
        tokenize_seq(seq, word_max_len, tokenizer) 
        for seq in tqdm.tqdm(df['code_name'])
        ]
    df['input_ids'] =pd.Series([token['input_ids'] for token in token_tmp])
    df['token_type_ids'] =pd.Series([token['token_type_ids'] for token in token_tmp])
    df['attention_mask'] =pd.Series([token['attention_mask'] for token in token_tmp])
    
    # desceemb input save
    np.save(os.path.join(save_path, f'input_ids_{value_mode}.npy'), np.array(df['input_ids'])) 
    np.save(os.path.join(save_path, f'token_type_ids_{value_mode}.npy'), np.array(df['token_type_ids'])) 
    np.save(os.path.join(save_path, f'attention_mask_{value_mode}.npy'), np.array(df['attention_mask'])) 

    if value_mode == 'VC':
        #Value for Value Concat(VC)
        value = np.array([df['value']])
        np.save(os.path.join(save_path, 'value.npy'), value[0])

    if value_mode =='DSVA':
        # Digit place embedding(DPE)
        df = digit_place_embed(df, tokenizer)
        np.save(os.path.join(save_path, f'input_ids_DSVA_DPE.npy'), np.array(df['input_ids'])) 
        np.save(os.path.join(save_path, f'token_type_ids_DSVA_DPE.npy'), np.array(df['token_type_ids'])) 
        np.save(os.path.join(save_path, f'attention_mask_DSVA_DPE.npy'), np.array(df['attention_mask']))            



def pooled_data_generation(
        dest_path, value_mode_list, seed_list, task_list, data_type
        ):
    save_path = os.path.join(dest_path, 'pooled')
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'label'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'fold'), exist_ok=True)
    
    text_input_list = ['input_ids', 'token_type_ids', 'attention_mask']
    print('pooled data generation')
    mimic_df = pd.read_pickle(os.path.join(dest_path, 'mimiciii_df.pkl'))
    eicu_df = pd.read_pickle(os.path.join(dest_path, 'eicu_df.pkl'))
    pooled_df = pd.concat([mimic_df, eicu_df], ignore_index=True)
    pooled_df.to_pickle(os.path.join(dest_path, 'pooled_df.pkl'))
    
    print('pooled fold generation')
    for seed in seed_list:
        mimic_split = pd.read_csv(os.path.join(dest_path, 'mimiciii', 'fold', f'fold_split_{seed}.csv'))
        eicu_split = pd.read_csv(os.path.join(dest_path, 'eicu', 'fold', f'fold_split_{seed}.csv'))
        pooled_split = pd.concat([mimic_split, eicu_split], ignore_index=True)
        pooled_split.to_csv(
            os.path.join(dest_path, 'pooled', 'fold', f'fold_split_{seed}.csv'), index=False
            )
    print('pooled label generation')    
    for task in task_list:
        mimic_label = np.load(os.path.join(dest_path, 'mimiciii', 'label', f'{task}.npy'))
        eicu_label = np.load(os.path.join(dest_path, 'eicu', 'label', f'{task}.npy'))
        pooled_label = np.concatenate([mimic_label, eicu_label], axis=0)
        np.save(os.path.join(dest_path, 'pooled', 'label', f'{task}.npy'), pooled_label)
    
    print('pooled text and code input generation')
    for value_mode in value_mode_list:
        for target in text_input_list:
            mimic_text_input = np.load(
                os.path.join(dest_path, 'mimiciii', f'{target}_{value_mode}.npy'),
                allow_pickle=True
                )
            eicu_text_input = np.load(
                os.path.join(dest_path, 'eicu', f'{target}_{value_mode}.npy'),
                allow_pickle=True
                )
            if data_type == 'predict':
                pooled_text_input = np.concatenate(
                    [mimic_text_input, eicu_text_input], axis=0
                    )
            elif data_type =='pretrain':
                mimic_text_input = mimic_text_input[0]
                eicu_text_input = eicu_text_input[0]

                # Calculating the maximum sequence length from both tensors
                dim_align = max(mimic_text_input.shape[1], eicu_text_input.shape[1])

                # Padding the tensors to have the same sequence length
                eicu_text_input = torch.nn.functional.pad(
                    eicu_text_input, (0, dim_align - eicu_text_input.shape[1])
                    )
                mimic_text_input= torch.nn.functional.pad(
                    mimic_text_input, (0, dim_align - mimic_text_input.shape[1])
                    )
                
                pooled_text_input = np.array(torch.cat(
                    [mimic_text_input, eicu_text_input], axis=0
                    )
                )          
            np.save(
                os.path.join(dest_path, 'pooled', f'{target}_{value_mode}.npy'), 
                pooled_text_input
                )
        
        mimic_vocab_path = os.path.join(
            dest_path, 'mimiciii', f'code_index_{value_mode}_vocab.pkl'
            )
        
        with open(mimic_vocab_path, 'rb') as file:
                mimic_vocab_dict = pickle.load(file)

        mimic_code_input = np.load(os.path.join(
            dest_path, 'mimiciii', f'code_index_{value_mode}.npy'), 
            allow_pickle=True
            )        
        eicu_code_input = np.load(os.path.join(
            dest_path, 'eicu', f'code_index_{value_mode}.npy'),
            allow_pickle=True
            )    

        # 0 = [PAD], 1 = [CLS], 2=[MASK]
        eicu_code_input = np.where(
            np.isin(eicu_code_input, [0, 1, 2], invert=True),
            eicu_code_input + len(mimic_vocab_dict),
            eicu_code_input
        )

        # pooled_code_input
        pooled_code_input = np.concatenate(
            [mimic_code_input, eicu_code_input], axis=0
            )
        
        np.save(os.path.join(
            dest_path, 'pooled', f'code_index_{value_mode}.npy'), 
                pooled_code_input
                )

    # seq_len 
    mimic_seq_len = np.load(os.path.join(dest_path, 'mimiciii', 'seq_len.npy'))
    eicu_seq_len = np.load(os.path.join(dest_path, 'eicu', 'seq_len.npy'))
    pooled_seq_len = np.concatenate([mimic_seq_len, eicu_seq_len], axis=0)
    np.save(os.path.join(dest_path, 'pooled', 'seq_len.npy'), pooled_seq_len)
    
    if data_type == 'predict':
        # value
        mimic_value = np.load(os.path.join(dest_path, 'mimiciii', 'value.npy'), allow_pickle=True)
        eicu_value = np.load(os.path.join(dest_path, 'eicu', 'value.npy'), allow_pickle=True)
        pooled_value = np.concatenate([mimic_value, eicu_value], axis=0)
        np.save(os.path.join(dest_path, 'pooled', 'value.npy'), pooled_value)
    
    print('pooled data generation has been done.')    
  