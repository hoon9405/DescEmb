from create_cohort import create_cohort
from dataframe_gen import preprocess
from numpy_convert import convert2numpy, pooled_data_generation
from preprocess_utils import label_npy_file, train_valid_test_split
import os
import argparse

# temp
import pandas as pd

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dest_path', type=str)
    parser.add_argument('--ccs_dx_tool_path', type=str)
    parser.add_argument('--icd10to9_path', type=str)
    parser.add_argument('--src_data', type=str, choices=['mimiciii', 'eicu', 'pooled'])
    parser.add_argument('--input_tables', nargs="+", 
                        choices=['lab', 'med', 'inf'], default=['lab', 'med', 'inf']
                        )
    parser.add_argument('--target_tasks', nargs="+", 
                        choices=['mortality', 'readmission', 'los_3day', 'los_7day', 'diagnosis'], 
                        default=['mortality', 'readmission', 'los_3day', 'los_7day', 'diagnosis']
                        )
    parser.add_argument('--seeds', nargs="+", 
                        default=[1])
    parser.add_argument('--ratio', nargs="+",
                        choices=['10', '30', '50', '70', '90', '100'], 
                        default=['10', '30', '50', '70', '90', '100'])
    parser.add_argument('--num_folds_split', type=int, default=5)
    parser.add_argument('--event_max_length', type=int, default=150)
    parser.add_argument('--event_min_length', type=int, default=5)
    parser.add_argument('--observe_window_time',  type=int, default=12)
    parser.add_argument('--min_stay_hours', type=int, default=12)
    parser.add_argument('--value_mode', type=str, choices=['NV', 'VA', 'DSVA', 'VC'], default=None)

    parser.add_argument('--data_type', type=str, choices=['predict', 'pretrain'], default='predict')
    parser.add_argument('--debug', action='store_true')
    return parser

def main():
    args = get_parser().parse_args()
    # file names
    csv_files = {
        'mimiciii':
                {
                'lab':'LABEVENTS', 
                'med':'PRESCRIPTIONS',
                'inf': 'INPUTEVENTS'
                },
        'eicu':
                {
                'lab':'lab', 
                'med':'medication',
                'inf':'infusionDrug'
                }
                            
    }

    # columns_map
    columns_mapping = {
        'mimiciii':
            {
            'LABEVENTS':
                {
                'HADM_ID':'ID',
                'CHARTTIME':'code_time',
                'ITEMID':'code_name',
                'VALUENUM':'value',
                'VALUEUOM':'uom',
                'FLAG':'issue'
                },
        'PRESCRIPTIONS':
            {
            'HADM_ID':'ID',
            'STARTDATE':'code_time',
            'DRUG':'code_name', 
            'ROUTE':'route', 
            'PROD_STRENGTH':'prod',
            'DOSE_VAL_RX':'value',
            'DOSE_UNIT_RX':'uom',
            },                                      
        'INPUTEVENTS': 
            {
            'HADM_ID':'ID',
            'CHARTTIME':'code_time', 
            'ITEMID':'code_name',
            'RATE':'value', 
            'RATEUOM':'uom',
            'STOPPED':'issue'
            },
        },
        'eicu':
            {
            'lab':
                {
                'patientunitstayid':'ID', 
                'labresultoffset':'code_offset',
                'labname':'code_name',
                'labresult':'value',
                'labmeasurenamesystem':'uom'
                },
            'medication':
                {
                'patientunitstayid':'ID',
                'drugstartoffset':'code_offset',
                'drugname':'code_name', 
                'routeadmin':'route',
                'ordercancelled':'issue'
                },      
            'infusionDrug':
                {
                'patientunitstayid':'ID',
                'infusionoffset':'code_offset',
                'drugname':'code_name',
                'infusionrate':'value'
                }
        }
    }

    # definition file name        
    def_file = {
        'mimiciii':
        {
        'LABEVENTS':'D_LABITEMS', 
        'INPUTEVENTS':'D_ITEMS', 
        },
        'eicu': None
        }
    
    issue_map = {
        'mimiciii':{
            'LABEVENTS': 
            ['abnormal'],                            
        'INPUTEVENTS':
            [
            'Restart',
            'NotStopd', 
            'Rewritten', 
            'Changed', 
            'Paused', 
            'Flushed', 
            'Stopped'
            ] ,
        'medication': 
            ['Yes'],          
        },
        'eicu': None
    }

    wd = os.getcwd()
    print('working directory .. : ', wd)
    
    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)
        print('create dest path..', args.dest_path)
    
    print("Destination directory is set to {}".format(args.dest_path))
    
    if args.value_mode is None:
            args.value_mode = ['NV', 'VA', 'DSVA', 'VC']
    else:
        args.value_mode = [args.value_mode]

    # check directory        
    if args.data_type =='pretrain':
        args.dest_path = os.path.join(args.dest_path, 'mlm')
    
    # create cohort
    if args.src_data in ['mimiciii', 'eicu']:
        create_cohort(
            args.src_data, 
            args.dataset_path, 
            args.dest_path, 
            args.ccs_dx_tool_path, 
            args.icd10to9_path, 
            args.min_stay_hours
        )

        preprocess(
            args.dataset_path, 
            args.dest_path,
            args.src_data,
            args.input_tables,
            csv_files[args.src_data], 
            columns_mapping[args.src_data], 
            issue_map[args.src_data], 
            def_file[args.src_data],
            args.event_max_length,
            args.event_min_length,
            args.data_type,
            args.debug
        )
        
        label_npy_file(args.dest_path, args.src_data, args.target_tasks)
        train_valid_test_split(
            args.dest_path, args.target_tasks, args.src_data, 
            args.seeds, args.ratio, args.num_folds_split
            )
            
        for mode in args.value_mode:
            convert2numpy(
                args.dest_path, 
                args.src_data, 
                mode, 
                args.data_type, 
                )
    
    if args.src_data =='pooled':
        pooled_data_generation(
            args.dest_path, 
            args.value_mode, 
            args.seeds, 
            args.target_tasks,
            args.data_type
            ) 
        
    print(f'preprocessing for {args.src_data} has been done.')

if __name__ == '__main__':
    main()