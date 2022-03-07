import pandas as pd
import numpy as np
import os
import datetime
import re
import random
import tqdm

def mimic_inf_merge(file, input_path, src):
    df_inf_mv_mm = pd.read_csv(os.path.join(input_path, src,'INPUTEVENTS_MV'+'.csv'), skiprows=lambda i: i>0 and random.random() > 0.01)
    df_inf_cv_mm = pd.read_csv(os.path.join(input_path, src, 'INPUTEVENTS_CV'+'.csv'), skiprows=lambda i: i>0 and random.random() > 0.01)
    df_inf_mv_mm['CHARTTIME'] = df_inf_mv_mm['STARTTIME']
    df_inf_mm = pd.concat([df_inf_mv_mm, df_inf_cv_mm], axis=0).reset_index(drop=True)
    print('mimic INPUTEVENTS merge!') 
    
    return df_inf_mm


def eicu_med_revise(file, input_path, src):
    df = pd.read_csv(os.path.join(input_path, src, file+'.csv'), skiprows=lambda i: i>0 and random.random() > 0.01)
    df['split'] = df['dosage'].apply(lambda x: str(re.sub(',', '',str(x))).split())
    def second(x):
        try:
            if len(pd.to_numeric(x))>=2:
                x = x[1:]
            return x
        except ValueError:
            return x

    df['split'] = df['split'].apply(second).apply(lambda s:' '.join(s))
    punc_dict = str.maketrans('', '', '.-')
    df['uom'] = df['split'].apply(lambda x: re.sub(r'[0-9]', '', x))
    df['uom'] = df['uom'].apply(lambda x: x.translate(punc_dict)).apply(lambda x: x.strip())
    df['uom'] = df['uom'].apply(lambda x: ' ' if x=='' else x)
    
    def hyphens(s):
        if '-' in str(s):
            s = str(s)[str(s).find("-")+1:]
        return s
    df['value'] = df['split'].apply(hyphens)
    df['value'] = df['value'].apply(lambda x: [float(s) for s in re.findall(r'-?\d+\.?\d*', x)])
    df['value'] = df['value'].apply(lambda x: x[-1] if len(x)>0 else x)
    df['value'] = df['value'].apply(lambda d: str(d).replace('[]',' '))
    
    return df


def eicu_inf_revise(file, input_path, src):
    df = pd.read_csv(os.path.join(input_path, src, file+'.csv'), skiprows=lambda i: i>0 and random.random() > 0.01 )
    df['split'] = df['drugname'].apply(lambda x: str(x).rsplit('(', maxsplit=1))
    def addp(x):
        if len(x)==2:
            x[1] = '(' + str(x[1])
        return x

    df['split'] = df['split'].apply(addp)
    df['split']=df['split'].apply(lambda x: x +[' '] if len(x)<2 else x)

    df['drugname'] = df['split'].apply(lambda x: x[0])
    df['uom'] = df['split'].apply(lambda x: x[1])
    df['uom'] = df['uom'].apply(lambda s: s[s.find("("):s.find(")")+1])

    toremove = ['()','', '(Unknown)', '(Scale B)', '(Scale A)',  '(Human)', '(ARTERIAL LINE)']

    df['uom'] = df['uom'].apply(lambda uom: ' ' if uom in toremove else uom)
    df = df.drop('split',axis=1)
    
    testing = lambda x: (str(x)[-1].isdigit()) if str(x)!='' else False
    code_with_num = list(pd.Series(df.drugname.unique())[pd.Series(df.drugname.unique()).apply(testing)==True])
    add_unk = lambda s: str(s)+' [UNK]' if s in code_with_num else s
    df['drugname'] = df['drugname'].apply(add_unk)
    
    return df

def column_rename(df, columns_map):
    df = df.rename(columns_map, axis='columns')

    return df

def issue_delete(df, csv_file, issue_map):
    if 'issue' in df.columns:
        issue_label = issue_map[csv_file]
        df.drop(df[df['issue'].isin(issue_label)].index, inplace=True)

    return df


def name_dict(df, csv_file, input_path, src, mimic_def_file):
    if csv_file in mimic_def_file:
        dict_name= mimic_def_file[csv_file]
        dict_path = os.path.join(input_path, src, dict_name+'.csv')
        code_dict = pd.read_csv(dict_path)
        key = code_dict['ITEMID']
        value = code_dict['LABEL']
        code_dict = dict(zip(key,value))
        df['code_name'] = df['code_name'].map(code_dict)

    return df


def null_convert(df):
    df = df.fillna(' ')

    return df


def ID_filter(df_icu, df):
    return df[df['ID'].isin(df_icu['ID'])]


def time_filter(df_icu, df, source, data_type):
    time_delta = datetime.timedelta(hours=12)
    if source =='mimic': 
        df = pd.merge(df, df_icu[['ID', 'INTIME', 'OUTTIME']], how='left', on='ID')
        df = df[df['code_time']!=' ']
        for col_name in ['code_time', 'INTIME', 'OUTTIME']:
            df[col_name] = pd.to_datetime(df[col_name])
        if data_type =='MICU':  
            df['INTIME+12hr'] = df['INTIME'] + time_delta
            df = df[(df['code_time']> df['INTIME']) & (df['code_time'] < df['OUTTIME']) & (df['code_time'] < df['INTIME+12hr'])]
        elif data_type =='TotalICU':
            df = df[(df['code_time']> df['INTIME']) & (df['code_time'] < df['OUTTIME'])]
 
        df['code_offset'] = df['code_time'] - df['INTIME']
        df['code_offset'] = df['code_offset'].apply(lambda x : x.seconds//60, 4)

    elif source =='eicu':
        if data_type =='MICU':
            df = df[(df['code_offset']> 0 ) | (df['code_offset'] < 12*60)]    
        elif data_type =='TotalICU':
            df = df[df['code_offset']> 0]   
    return df            

  
def min_length(df, min_length):
    df = df[df['code_name'].map(type) ==list]
    df['code_length'] = df['code_name'].map(len)
    df = df[df['code_length']>=min_length]
    df = df.drop('code_length', axis=1)

    return df


def offset2order(offset_seq):
    offset_set = set(offset_seq)

    dic = {}
    for idx, offset in enumerate(list(offset_set)):
        dic[offset] = idx
    
    def convert(x):
        return dic[x]
    
    order_seq = list(map(convert, offset_seq))

    return order_seq

 
def text2idx(seq, vocab):
    def convert(x):
        return vocab[x]
    
    return seq.apply(lambda x : convert(x))


def merge_df(df_lab, df_med, df_inf):
    df_merge = pd.concat([df_lab, df_med, df_inf], axis=0)
    df_merge = df_merge[['ID', 'code_name', 'value', 'uom', 'code_offset']]
        
    return df_merge 


def list_prep(df, df_icu):
    column_list = ['code_name', 'code_offset', 'value', 'uom']
    df_agg = df.groupby(['ID']).agg({column: lambda x: x.tolist() for column in column_list})
    df = pd.merge(df_icu, df_agg, how='left', on=['ID'])

    return df


def making_vocab(df):
    vocab_dict = {}
    vocab_dict['[PAD]'] = 0
      
    df['merge_code_set'] = df['code_name'].apply(lambda x : list(set(x)))
    vocab_set = []
    for codeset in df['merge_code_set']:
        vocab_set.extend(codeset) 
    vocab_set = list(set(vocab_set))
    for idx, vocab in enumerate(vocab_set):
        vocab_dict[vocab] = idx+1
        
    return vocab_dict


def ID_rename(df_icu, src):
    if src =='mimic' : 
        icu_ID = 'HADM_ID'
    elif src=='eicu':
        icu_ID = 'patientunitstayid'
        
    df_icu['ID'] = df_icu[icu_ID]
    df_icu = df_icu.drop(columns=icu_ID)

    return df_icu

def pad(sequence, max_length):
    if len(sequence) > max_length:

        return sequence[:max_length]
    else:
        pad_length = max_length-len(sequence)
        zeros = list(np.zeros(pad_length))
        sequence.extend(zeros) 

        return sequence

def sampling(sequence, walk_len, max_length):
    seq_len = len(sequence)
    seq_index_start = [i*walk_len  for i in range(((seq_len-max_length)//walk_len)+1)]
    
    return [sequence[i:(i+max_length)] for i in seq_index_start]

def sortbyoffset(df):
    print('sortbyoffset')
    sorted = df.sort_values(['code_offset'],ascending=True).groupby('ID')
    return sorted

def preprocess(input_path,
                item_list,
                csv_files_dict, 
                columns_map_dict, 
                issue_map, 
                mimic_def_file,
                max_length,
                data_type):

    for src in ['mimic', 'eicu']:
        df_icu = pd.read_pickle(os.path.join(input_path, src, f'{src}_cohort.pk'))
        df_icu = ID_rename(df_icu, src)
        for item in item_list:
            print('data preparation initialization .. {} {}'.format(src, item))
            file = csv_files_dict[src][item]
            columns_map = columns_map_dict[src][file] # the files from mimic that we want
            if src =='mimic' and item =='inf':
                df = mimic_inf_merge(file, input_path, src)
            elif src=='eicu' and item=='med':
                df = eicu_med_revise(file, input_path, src)
            elif src=='eicu' and item=='inf':
                df = eicu_inf_revise(file, input_path, src)
            else:
                df = pd.read_csv(os.path.join(input_path, src, file+'.csv'))
            print('df_load ! .. {} {}'.format(src, item))

            df = column_rename(df, columns_map)
            df = issue_delete(df, file, issue_map)
            df = name_dict(df, file, input_path, src, mimic_def_file)
            df = null_convert(df)
            df = ID_filter(df_icu, df)
            df = time_filter(df_icu, df, src, data_type)

            if item == 'lab':
                lab = df.copy()
            elif item =='med':
                med = df.copy()
                #med = med_align(src, med)
            elif item =='inf':
                inf = df.copy()

        del(df)
        print('data preparation finish for three items \n second preparation start soon..')

        df = merge_df(lab, med ,inf)
        print('lab med inf three categories merged in one!')
        # breakpoint()
        df = sortbyoffset(df)
        breakpoint()
        df = list_prep(df, df_icu)
        df = min_length(df, 5).reset_index(drop=True)
        df['code_order'] = df['code_offset'].map(lambda x : offset2order(x))  
        # sequence align with offset order

        df['seq_len'] = df['code_name'].map(len)
        #pad
       
        column_list = ['code_name', 'code_offset', 'value', 'uom', 'code_order']
        if data_type == 'MICU':
            for column in column_list:
                df[column] = df[column].map(lambda x : pad(x, max_length))
        
        elif data_type == 'TotalICU':
            df_short = df[df['seq_len'] <= max_length].reset_index(drop=True)
            df_long = df[df['seq_len'] > max_length].reset_index(drop=True)
           
            for i, column in enumerate(column_list):
                df_short[column] = df[column].map(lambda x : pad(x, max_length))  
                df_long[column] = df[column].map(lambda x: sampling(x, max_length//3, max_length))

            df_long = df_long.explode(column_list).reset_index(drop=True)
            df = pd.concat([df_short, df_long], axis=0).reset_index(drop=True)
            del df_short, df_long

        print('Preprocessing completed.')    
        print('Writing', '{}_df.pkl'.format(src), 'to', input_path)
        df.to_pickle(os.path.join(input_path,'{}_df.pkl'.format(src)))
    
    df_mm = pd.read_pickle(os.path.join(input_path,'mimic_df.pkl'.format(src)))
    df_ei = pd.read_pickle(os.path.join(input_path,'eicu_df.pkl'.format(src)))
    df_pooled = pd.concat((df_mm,df_ei), axis=0).reset_index(drop=True)
    df_pooled.to_pickle(os.path.join(input_path,'pooled_df.pkl'.format(src)))
    del df_mm, df_ei, df_pooled                  


        
