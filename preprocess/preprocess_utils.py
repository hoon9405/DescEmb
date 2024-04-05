import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
import pandas as pd

def multi_hot(column):
    mlb = MultiLabelBinarizer()
    encoded_dx = mlb.fit_transform(column)
    multihot_vector=mlb.classes_
    print(multihot_vector)
    df_dx = pd.DataFrame(encoded_dx, columns=multihot_vector)
    df_dx.columns = df_dx.columns.astype(np.int16)
    df_dx = df_dx.sort_index(axis=1)
    return df_dx

def label_npy_file(dest_path, src_data, target_tasks):
    save_path = os.path.join(dest_path, src_data, 'label')
    os.makedirs(save_path, exist_ok=True)
    
    df_pickle_name = f'{src_data}_df.pkl'
    df = pd.read_pickle(os.path.join(dest_path, df_pickle_name))
    # target tasks labeling
    for task in target_tasks:
        column = df[task]
        if task =='diagnosis':
            column = multi_hot(column)
            if column.shape[1] == 17:
                zeros = pd.Series(list(np.zeros(column.shape[0], dtype='int16')))
                column[15]= zeros
                column = column.reindex(sorted(column.columns), axis=1)
        array = np.array(column, dtype='int16')
        save_name = f'{task}.npy'
        
        print('label numpy file save to ', os.path.join(save_path, save_name))
        np.save(os.path.join(save_path, save_name), array)


def train_valid_test_split(dest_path, target_tasks, src_data, seeds, ratio_list, num_folds_split):
    save_path = os.path.join(dest_path, src_data, 'fold')
    os.makedirs(save_path, exist_ok=True)

    df = pd.read_pickle(os.path.join(dest_path, f'{src_data}_df.pkl'))
    
    for seed in seeds:
        print('seed : ', seed)
        sss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=len(df)//num_folds_split, random_state=seed)
        mskf = MultilabelStratifiedKFold(n_splits=num_folds_split, random_state=seed, shuffle=True)
        X = df[['ID'] + target_tasks].copy()
        # Binary classification Stratified Kfold Split
        for task in target_tasks:
            if task == 'diagnosis':
                continue
            print(f'{task} train and test split')
            fold_task = f'{task}_fold'
            X[fold_task]=1
            print('X fold_task value counts \n',X[fold_task].value_counts())
            y = X[task]

            #train / test split 4:1
            for train_index, test_index in sss_train_test.split(X, y):
                X.loc[test_index, fold_task] = 0  

            # Splitting train into train/valid 4:1 
            for train_index, valid_index in sss_train_test.split(X.loc[train_index], y.loc[train_index]):
                X.loc[valid_index, fold_task] = 2

            # Now resetting index once after all operations are done
            X = X.reset_index(drop=True)
            
            for fold in X[fold_task].unique():
                print(f"\n{fold} label distribution:")
                print(X[X[fold_task] == fold][task].value_counts(normalize=True))
            print(f'fold split {src_data} with {task} done \n', X[fold_task].value_counts())

        # Muli_label stratified Kfold Split    
        if 'diagnosis' in target_tasks:
            print('diagnosis multi label stratified split') 
            X['diagnosis_fold'] = 1
            
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(X['diagnosis'])
            # dummy X
            X_dummy = np.random.rand(y.shape[0], y.shape[1])
            
            # Split the dataset into initial train and test sets
            initial_train_index, test_index = next(
                mskf.split(X_dummy, y)
                )
            X.loc[test_index, 'diagnosis_fold'] = 0  # Assign 0 to test fold

            # Further split the initial train set into final train and validation sets
            train_index, valid_index = next(
                mskf.split(X_dummy[initial_train_index], y[initial_train_index])
                )
            X.loc[initial_train_index[valid_index], 'diagnosis_fold'] = 2  # Assign 2 to validation fold

            # Now that we have our folds, we reset the index to organize our DataFrame
            X = X.reset_index(drop=True)
            X.to_csv(os.path.join(
                dest_path, f'{src_data}', 'fold', f'fold_split_{seed}.csv'), index=False
                )
            # Verifying the split by printing the distribution of each fold
            print('Diagnosis multi-label stratified split results:')
            fold_counts = X['diagnosis_fold'].value_counts().sort_index()
            print(fold_counts)
            for fold in fold_counts.index:
                print(f"\nFold {fold} label distribution:")
                print(
                    X[X['diagnosis_fold'] == fold]['diagnosis'].value_counts(normalize=True)
                )
        print('preparing split for few-shot learning')

        for ratio in ratio_list:
            print(f"Ratio: {ratio}")
            for task in target_tasks:
                fold_task = f'{task}_fold'
                print(task, ratio)
                X, num_exclude_train, num_exclude_valid = split_and_update(X, task, fold_task, ratio, seed)
                print(f"{task}: Excluded {num_exclude_train} samples from train, {num_exclude_valid} samples from valid")

            for task in target_tasks:
                fold_task = f'{task}_fold'
                print_label_distribution(X, task, fold_task, ratio)
        
        X.to_csv(os.path.join(
            dest_path, f'{src_data}', 'fold', f'fold_split_{seed}.csv'), index=False
            )
        
        
def split_and_update(X, task, fold_task, ratio, seed):
    # 분할할 데이터 필터링
    X[fold_task+f'_{ratio}'] = X[fold_task]
    if ratio =='100':
        return X, 0, 0
    
    filter_mask = X[fold_task] == 1
    filtered_data = X[filter_mask]

    _, exclude_data_train = train_test_split(
        filtered_data, test_size=1-(int(ratio)/100), 
        stratify=None, 
        random_state=seed
        )
    X.loc[exclude_data_train.index, fold_task+f'_{ratio}'] = -1
    
    filter_mask_valid = X[fold_task] == 2
    filtered_data_valid = X[filter_mask_valid]
    _, exclude_data_valid = train_test_split(
        filtered_data_valid, test_size=1-(int(ratio)/100), 
        stratify=None, 
        random_state=seed
        )

    X.loc[exclude_data_valid.index, fold_task+f'_{ratio}'] = -1

    return X, len(exclude_data_train), len(exclude_data_valid)


def print_label_distribution(X, task, fold_task, ratio):
    print(f"\n{task} label distribution:")
    for fold_type, fold_value in [("Train set", 1), ("Valid set", 2)]:
        print(f"{fold_type}:")
        print(X[X[fold_task+f'_{ratio}'] == fold_value][task].value_counts(normalize=True))