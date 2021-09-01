import pandas as pd
import numpy as np
import os
import sys
import pickle
import datetime
import warnings
import easydict
import re
import random

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

def issue_delete(df, csv_file):
    if 'issue' in df.columns:
        issue_label = issue_map[csv_file]
        df.drop(df[df['issue'].isin(issue_label)].index, inplace=True)

    return df


def name_dict(df, csv_file, input_path, src):
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


def time_filter(df_icu, df, source):
    time_delta = datetime.timedelta(hours=12)
    if source =='mimic': 
        df = pd.merge(df, df_icu[['ID', 'INTIME', 'OUTTIME']], how='left', on='ID')
        df = df[df['code_time']!=' ']
        for col_name in ['code_time', 'INTIME', 'OUTTIME']:
            df[col_name] = pd.to_datetime(df[col_name])
        df['INTIME+12hr'] = df['INTIME'] + time_delta
        df = df[(df['code_time']> df['INTIME']) | (df['code_time'] < df['OUTTIME']) | (df['code_time'] < df['INTIME+12hr'])]
        df['code_offset'] = df['code_time'] - df['INTIME']
        df['code_offset'] = df['code_offset'].apply(lambda x : x.seconds//60, 4)
    elif source =='eicu':
        df = df[(df['code_offset']> 0 ) | (df['code_offset'] < 12*60)]    

    return df            


def pad(sequence, max_length = 150):
    if len(sequence) > 150:

        return sequence[:150]
    else:
        pad_length = 150-len(sequence)
        zeros = list(np.zeors(pad_length))
        sequence.extend(zeros)

        return sequence
    

def min_length(df):
    df = df[df['code_name'].map(type) ==list]
    df['code_length'] = df['code_name'].map(len)
    print
    df = df[df['code_length']>=5]
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


def preprocess(args):
    for src in args.source_list:
        df_icu = pd.read_pickle(os.path.join(args.data_input_path, src, f'{src}_cohort.pk'))
        df_icu = ID_rename(df_icu, src)
        for item in args.item_list:
            print('data preparation initialization .. {} {}'.format(src, item))
            file = csv_files_dict[src][item]
            columns_map = columns_map_dict[src][file] # the files from mimic that we want
            if src =='mimic' and item =='inf':
                df = mimic_inf_merge(file, args.data_input_path, src)
            elif src=='eicu' and item=='med':
                df = eicu_med_revise(file, args.data_input_path, src)
            elif src=='eicu' and item=='inf':
                df = eicu_inf_revise(file, args.data_input_path, src)
            else:
                df = pd.read_csv(os.path.join(args.data_input_path, src, file+'.csv'), skiprows=lambda i: i>0 and random.random() > 0.01)
            print('df_load ! .. {} {}'.format(src, item))
            
            df = column_rename(df, columns_map)
            df = issue_delete(df, file)
            df = name_dict(df, file, args.data_input_path, src)
            df = null_convert(df)
            df = ID_filter(df_icu, df)
            df = time_filter(df_icu, df, src)
                        
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
        df = list_prep(df, df_icu)
        df = min_length(df)
        
        df['code_order'] = df['code_offset'].map(lambda x : offset2order(x))   
        vocab_dict = making_vocab(df)
        print('Generated vocabulary of length', len(vocab_dict), '\n')

        print('Preprocessing completed.')
        
        print('Writing', '{}_df.pkl'.format(src), 'to', args.data_output_path)
        df.to_pickle(os.path.join(args.data_output_path,'{}_df.pkl'.format(src)))

        print('Writing', '{}_vocab.pkl'.format(src), 'to', args.data_output_path)
        with open('{}/{}_vocab.pkl'.format(args.data_output_path, src), 'wb') as f:
            pickle.dump(vocab_dict, f)
        

# file names

mimic_csv_files = {'lab':'LABEVENTS', 'med':'PRESCRIPTIONS',
                        'inf': 'INPUTEVENTS'}

eicu_csv_files = {'lab':'lab', 'med':'medication','inf':'infusionDrug'}
                  
    
# definition file name        
mimic_def_file = {'LABEVENTS':'D_LABITEMS', 
                         'INPUTEVENTS_CV':'D_ITEMS', 'INPUTEVENTS_MV':'D_ITEMS'}

# columns_map
mimic_columns_map = {'LABEVENTS':
                         {'HADM_ID':'ID','CHARTTIME':'code_time','ITEMID':'code_name',
                          'FLAG':'issue'},
                     'PRESCRIPTIONS':
                         {'HADM_ID':'ID','STARTDATE':'code_time',
                          'DRUG':'code_name', 'ROUTE':'route', 'PROD_STRENGTH':'prod'},                                      
                      'INPUTEVENTS': 
                         {'HADM_ID':'ID','CHARTTIME':'code_time', 
                          'ITEMID':'code_name','RATE':'value', 'RATEUOM':'uom',
                          'STOPPED':'issue'}}

eicu_columns_map =  {'lab':
                         {'patientunitstayid':'ID', 'labresultoffset':'code_offset','labname':'code_name'},
                     'medication':
                         {'patientunitstayid':'ID','drugstartoffset':'code_offset','drugname':'code_name', 
                          'routeadmin':'route',
                          'ordercancelled':'issue'},      
                      'infusionDrug':
                           {'patientunitstayid':'ID','infusionoffset':'code_offset', 'drugname':'code_name'}
                    }
# issue map
issue_map = {'LABEVENTS': ['abnormal'],                            
             'INPUTEVENTS':['Restart','NotStopd', 'Rewritten', 'Changed', 'Paused', 'Flushed', 'Stopped'] ,
             'medication': ['Yes'],          
            }

csv_files_dict = {'mimic':mimic_csv_files, 'eicu':eicu_csv_files}
columns_map_dict = {'mimic':mimic_columns_map, 'eicu':eicu_columns_map}


def main():
    wd = os.getcwd()
    print('working directory .. : ', wd)

    args = easydict.EasyDict({'data_input_path' : '/home/ghhur/data/csv',
                            'data_output_path' : '/home/ghhur/data/csv',
                            'source_list' : ['mimic', 'eicu'],
                            'item_list' : ['lab','med', 'inf'], 
                            'max_length' : 150,
                            'min_length' : 5,
                            'window_time' : 12,
                            'p' : 0.01})

    preprocess(args)

if __name__ == '__main__':
    main()


