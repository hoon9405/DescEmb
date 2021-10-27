from create_dataset import create_MIMIC_dataset, create_eICU_dataset
from dataframe_gen import preprocess
from numpy_convert import convert2numpy
from preprocess_utils import label_npy_file
import os
import argparse

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input_path', type=str)
    parser.add_argument('--data_output_path', type=str)
    parser.add_argument('--max_length', type=int, default=150)
    parser.add_argument('--min_length', type=int, default=5)
    parser.add_argument('--window_time',  type=int, default=12)
    parser.add_argument('--data_type', type=str, choices=['MICU', 'TotalICU'], default='MICU')
    return parser

def main():
    args = get_parser().parse_args()
    # file names
    mimic_csv_files = {'lab':'LABEVENTS', 
                        'med':'PRESCRIPTIONS',
                        'inf': 'INPUTEVENTS'}

    eicu_csv_files = {'lab':'lab', 
                    'med':'medication',
                    'inf':'infusionDrug'}

    # definition file name        
    mimic_def_file = {'LABEVENTS':'D_LABITEMS', 
                    'INPUTEVENTS_CV':'D_ITEMS', 
                    'INPUTEVENTS_MV':'D_ITEMS'}

    # columns_map
    mimic_columns_map = {'LABEVENTS':
                            {'HADM_ID':'ID',
                            'CHARTTIME':'code_time',
                            'ITEMID':'code_name',
                            'VALUENUM':'value',
                            'VALUEUOM':'uom',
                            'FLAG':'issue'
                            },
                        'PRESCRIPTIONS':
                            {'HADM_ID':'ID',
                            'STARTDATE':'code_time',
                            'DRUG':'code_name', 
                            'ROUTE':'route', 
                            'PROD_STRENGTH':'prod',
                            'DOSE_VAL_RX':'value',
                            'DOSE_UNIT_RX':'uom',
                            },                                      
                        'INPUTEVENTS': 
                            {'HADM_ID':'ID',
                            'CHARTTIME':'code_time', 
                            'ITEMID':'code_name',
                            'RATE':'value', 
                            'RATEUOM':'uom',
                            'STOPPED':'issue'
                            }
    }

    eicu_columns_map =  {'lab':
                            {'patientunitstayid':'ID', 
                            'labresultoffset':'code_offset',
                            'labname':'code_name',
                            'labresult':'value',
                            'labmeasurenamesystem':'uom'
                            },
                        'medication':
                            {'patientunitstayid':'ID',
                            'drugstartoffset':'code_offset',
                            'drugname':'code_name', 
                            'routeadmin':'route',
                            'ordercancelled':'issue'
                            },      
                        'infusionDrug':
                            {'patientunitstayid':'ID',
                            'infusionoffset':'code_offset',
                            'drugname':'code_name',
                            'infusionrate':'value'
                            }
    }
   
    issue_map = {'LABEVENTS': 
                    ['abnormal'],                            
                'INPUTEVENTS':
                    ['Restart',
                    'NotStopd', 
                    'Rewritten', 
                    'Changed', 
                    'Paused', 
                    'Flushed', 
                    'Stopped'
                    ] ,
                'medication': 
                    ['Yes'],          
                }

    csv_files_dict = {'mimic':mimic_csv_files, 
                        'eicu':eicu_csv_files
    }
    columns_map_dict = {'mimic':mimic_columns_map, 
                           'eicu':eicu_columns_map
    }
    item_list = ['lab','med', 'inf']
    wd = os.getcwd()
    print('working directory .. : ', wd)

    #create_MIMIC_dataset(os.path.join(args.data_input_path, 'mimic'))
    #create_eICU_dataset(os.path.join(args.data_input_path, 'eicu'))

    preprocess(args.data_input_path, 
                    item_list,
                   csv_files_dict, 
                   columns_map_dict, 
                   issue_map, 
                   mimic_def_file,
                   args.max_length,
                   args.data_type)
    
    #convert2numpy(args.data_input_path, args.data_output_path)
    #label_npy_file(args.data_input_path, args.data_output_path)
    
    print('preprocess finish!!')

if __name__ == '__main__':
    main()