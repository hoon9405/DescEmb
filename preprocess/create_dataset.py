import pandas as pd
import numpy as np
import os
import pickle
import warnings
import easydict
from collections import Counter

warnings.filterwarnings( 'ignore' )

# Create MIMIC-III dataset
def create_MIMIC_dataset(input_path):
    patient_path = os.path.join(input_path, 'PATIENTS.csv')
    icustay_path = os.path.join(input_path, 'ICUSTAYS.csv')
    dx_path = os.path.join(input_path, 'DIAGNOSES_ICD.csv')

    patients = pd.read_csv(patient_path)
    icus = pd.read_csv(icustay_path)
    dx = pd.read_csv(dx_path)
    print('length of PATIENTS.csv  : ', len(patients))
    print('length of ICUSTAYS.csv  : ', len(icus))
    print('length of DIAGNOSIS_ICD.csv  : ', len(icus))

    temp = icus[(icus['FIRST_CAREUNIT'] == icus['LAST_CAREUNIT'])]
    #temp = temp[temp.LAST_CAREUNIT == 'MICU']     
    #->For Total ICU
    
    temp = temp.drop(columns=['ROW_ID'])
    temp['INTIME'] = pd.to_datetime(temp['INTIME'], infer_datetime_format=True)
    temp['OUTTIME'] = pd.to_datetime(temp['OUTTIME'], infer_datetime_format=True)

    patients['DOB'] = pd.to_datetime(patients['DOB'], infer_datetime_format=True)
    patients['DOD'] = pd.to_datetime(patients['DOD'], infer_datetime_format=True)
    patients['DOD_HOSP'] = pd.to_datetime(patients['DOD_HOSP'], infer_datetime_format=True)
    patients['DOD_SSN'] = pd.to_datetime(patients['DOD_SSN'], infer_datetime_format=True)
    patients = patients.drop(columns=['ROW_ID'])

    small_patients = patients[patients.SUBJECT_ID.isin(temp.SUBJECT_ID)]
    temp = temp.merge(small_patients, on='SUBJECT_ID', how='left')

    datediff = np.array(temp.INTIME.dt.date) - np.array(temp.DOB.dt.date)
    age = np.array([x.days // 365 for x in datediff])
    temp['age'] = age
    temp = temp[temp.age >= 18]
    print('length of temp  :', len(temp))

    readmit = temp.groupby('HADM_ID')['ICUSTAY_ID'].count()
    readmit_labels = (readmit > 1).astype('int64').to_frame().rename(columns={'ICUSTAY_ID':'readmission'})
    print('readmission value counts :', readmit_labels.value_counts())

    small_temp = temp.loc[temp.groupby('HADM_ID').INTIME.idxmin()]
    readmission_cohort = small_temp.join(readmit_labels, on='HADM_ID')
    cohort = readmission_cohort
    dead = cohort[~pd.isnull(cohort.DOD_HOSP)].copy()
    dead_labels = ((dead.DOD_HOSP.dt.date > dead.INTIME.dt.date) & (dead.DOD_HOSP.dt.date <= dead.OUTTIME.dt.date))
    dead_labels = dead_labels.astype('int64')
    dead['mortality'] = np.array(dead_labels)
    cohort = cohort.merge(dead.iloc[:, [2,-1]], on='ICUSTAY_ID', how='left')
    cohort['mortality'] = cohort['mortality'].fillna(0)
    cohort = cohort.astype({'mortality': int})
    cohort['los_3day'] = (cohort['LOS'] > 3.).astype('int64')
    cohort['los_7day'] = (cohort['LOS'] > 7.).astype('int64')
    cohort12h = cohort[cohort['LOS'] > 0.5]
    cohort12hDx = dx[dx.HADM_ID.isin(cohort12h.HADM_ID)]
    diagnosis = cohort12hDx.groupby('HADM_ID')['ICD9_CODE'].apply(list).to_frame()
    tempdf = cohort12h.join(diagnosis, on='HADM_ID')

    tempdf['12h_obs'] = tempdf.INTIME + pd.Timedelta(pd.offsets.Hour(12))
    tempdf['24h_obs'] = tempdf.INTIME + pd.Timedelta(pd.offsets.Hour(24))

    cohort_mm = tempdf.reset_index(drop=True).copy()
    
    #diagnosis label
    # ccs_dx = pd.read_csv('/home/ghhur/DescEmb/preprocess/ccs_multi_dx_tool_2015.csv')
    # ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:].str[:-1].str.replace(" ", "")
    # ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:].str[:-1]
    # level1 = {}
    # for x, y in zip(ccs_dx["'ICD-9-CM CODE'"], ccs_dx["'CCS LVL 1'"]):
    #     level1[x] = y
    
    # dx1_list = []
    # for idx, dxx in enumerate(cohort_mm['ICD9_CODE']):
    #     one_list = []
    #     for dx in dxx:
    #         dx1 = level1[dx]
    #         one_list.append(dx1)
    #     dx1_list.append(list(set(one_list)))
    # cohort_mm['diagnosis'] = pd.Series(dx1_list)
    # cohort_mm = cohort_mm[cohort_mm['diagnosis'] !=float].reset_index(drop=True)
    # dx1_length = [len(i) for i in dx1_list]
    # print("average length: ", np.array(dx1_length).mean())
    # print('dx freqeuncy', np.bincount(dx1_length))
    # print("max length: ", np.array(dx1_length).max())
    # print("min length: ", np.array(dx1_length).min())

    #save as pickle
    pickle.dump(cohort_mm, open(os.path.join(input_path, 'mimic_cohort.pk'), 'wb'), -1)


# Create eICU dataset
def create_eICU_dataset(input_path):
    patient_path = os.path.join(input_path, 'patient.csv')
    patient_df = pd.read_csv(patient_path)
    dx_path = os.path.join(input_path, 'diagnosis.csv')
    dx = pd.read_csv(dx_path)

    print('Unique patient unit stayid : ', len(set(patient_df.patientunitstayid)))

    micu = patient_df
    #micu = patient_df[patient_df.unittype == 'MICU']  
    #-> For Total ICU

    null_index =micu[micu['age'].isnull()==True].index
    micu.loc[null_index, 'age'] = 1
    micu = micu.replace('> 89', 89)

    micu.loc[:, 'age'] = micu.loc[:, 'age'].astype('int')
    micuAge = micu[micu.age >= 18]

    readmit = micuAge.groupby('patienthealthsystemstayid')['patientunitstayid'].count()
    readmit_labels = (readmit > 1).astype('int64').to_frame().rename(columns={'patientunitstayid':'readmission'})
    firstIcus = micuAge.loc[micuAge.groupby('patienthealthsystemstayid').hospitaladmitoffset.idxmax()]
    readmission_cohort = firstIcus.join(readmit_labels, on='patienthealthsystemstayid')

    cohort = readmission_cohort

    cohort['mortality'] = (cohort['unitdischargestatus'] == 'Expired').astype('int64')
    cohort['losday'] = (cohort['unitdischargeoffset'].astype('float') / (24.*60.))
    cohort['los_3day'] = (cohort['losday'] > 3.).astype('int64')
    cohort['los_7day'] = (cohort['losday'] > 7.).astype('int64')

    cohort12h = cohort[cohort['unitdischargeoffset'] > 60*12]
    cohort12hDx = dx[dx.patientunitstayid.isin(cohort12h.patientunitstayid)]
    diagnosis = cohort12hDx.groupby('patientunitstayid')['diagnosisstring'].apply(list).to_frame()

    dxDict = dict(enumerate(cohort12hDx.groupby('diagnosisstring').count().index))
    dxDict = dict([(v,k) for k,v in dxDict.items()])
    print('diganosis unique : ', len(dxDict))

    tempdf = cohort12h.join(diagnosis, on='patientunitstayid') 
    cohort_ei = tempdf.copy().reset_index(drop=True)
    #cohort_ei = eicu_diagnosis_label(cohort_ei)
    #cohort_ei = cohort_ei[cohort_ei['diagnosis'] !=float].reset_index(drop=True)
    cohort_ei = cohort_ei.reset_index(drop=True)
    pickle.dump(cohort_ei, open(os.path.join(input_path, 'eicu_cohort.pk'), 'wb'), -1)



def eicu_diagnosis_label(eicu_cohort):
    ccs_dx = pd.read_csv('/home/ghhur/DescEmb/preprocess/ccs_multi_dx_tool_2015.csv')
    ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:].str[:-1].str.replace(" ", "")
    ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:].str[:-1]
    level1 = {}
    for x, y in zip(ccs_dx["'ICD-9-CM CODE'"], ccs_dx["'CCS LVL 1'"]):
        level1[x] = y 

    eicu_dx_df = eicu_cohort.dropna(subset=['diagnosisstring']).copy().reset_index(drop=True)
    eicu_diagnosis_list = []
    for x in eicu_dx_df['diagnosisstring']:
        eicu_diagnosis_list.extend(x)
    eicu_dx_unique = list(set(eicu_diagnosis_list))
    eicu_dx = pd.read_csv('/home/ghhur/data/csv/eicu/diagnosis.csv')

    # eicu_dx all diagnosis status
    eicu_dx_list = list(eicu_dx['icd9code'].values)
    eicu_dx_list = [x for x in eicu_dx_list if x != 'nan' and type(x) != float]
    eicu_dx_list = [y.strip().replace('.', '') for x in eicu_dx_list for y in x.split(',')]
    eicu_ids = list(eicu_dx_df['patientunitstayid'].values)

    # drop the icd9code NaN for right now
    eicu_dx = eicu_dx.dropna(subset=['icd9code']).copy().reset_index(drop=True)

    # make diagnosisstring - ICD9 code dictionary
    diagnosisstring_code_dict = {}
    key_error_list = []

    for index, row in eicu_dx.iterrows():
        diagnosis_string = row['diagnosisstring']
        icd9code = row['icd9code']
        icd9code = icd9code.split(',')[0].replace('.','')
        try: 
            eicu_level1 = level1[icd9code]
            diagnosisstring_code_dict[diagnosis_string] = eicu_level1
        except KeyError:
            key_error_list.append(diagnosis_string)  

    # Check key error list
    key_error_list = list(set(key_error_list))
    print('Number of diagnosis with only ICD 10 code: {}'.format(len(key_error_list)))

    # icd10 to icd9 mapping csv file
    icd10_icd9 = pd.read_csv('/home/ghhur/DescEmb/preprocess/icd10cmtoicd9gem.csv')

    # make icd10 - icd9 dictionary
    icd10_icd9_dict = {}
    for x, y in zip(icd10_icd9['icd10cm'], icd10_icd9['icd9cm']):
        icd10_icd9_dict[x] = y

    # map icd10 to icd9 code
    two_icd10_code_list = []
    icd10_key_error_list = []
    for i in range(len(key_error_list)):
        icd10code = eicu_dx[eicu_dx['diagnosisstring'] == key_error_list[i]]['icd9code'].values[0].split(',')
        if len(icd10code) >= 2:
            two_icd10_code_list.append(key_error_list[i])
            continue
            
        elif len(icd10code) == 1:
            icd10code = icd10code[0].replace('.','')
            try:
                icd9code = icd10_icd9_dict[icd10code]
                diagnosisstring_code_dict[key_error_list[i]] = level1[icd9code]
            except KeyError:
                icd10_key_error_list.append(key_error_list[i])
    print('Number of more than one icd10 codes : {}'.format(len(two_icd10_code_list)))
    print('Number of icd10key_error_list : {}'.format(len(icd10_key_error_list)))


    # deal with more than one ICD10 code
    class_list = ['6', '7', '6', '7', '2', '6', '6', '7', '6', '6','6']
    for i in range(11):
        diagnosisstring_code_dict[two_icd10_code_list[i]] = class_list[i]

    # fill in the blank!
    have_to_find = []
    already_in = []
    for i in range(len(eicu_dx_unique)):
        single_dx = eicu_dx_unique[i]
        try:
            oneoneone = diagnosisstring_code_dict[single_dx]
            already_in.append(single_dx)
        except KeyError:
            have_to_find.append(single_dx)
    print('Number of dx we have to find...{}'.format(len(have_to_find)))

    # one hierarchy above
    have_to_find2 = []
    for i in range(len(have_to_find)):
        s = "|".join(have_to_find[i].split('|')[:-1])
        try:
            depth1_code = diagnosisstring_code_dict[s]
            diagnosisstring_code_dict[have_to_find[i]] = depth1_code
        except KeyError:
            have_to_find2.append(have_to_find[i])
    print('Number of dx we have to find...{}'.format(len(have_to_find2)))

    # hierarchy below
    dict_keys = list(diagnosisstring_code_dict.keys())

    have_to_find3 = []
    for i in range(len(have_to_find2)):
        s = have_to_find2[i]
        dx_list = []
        for k in dict_keys:
            if k[:len(s)] == s:
                dx_list.append(diagnosisstring_code_dict[k])
        
        dx_list = list(set(dx_list))
        if len(dx_list) == 1:
            diagnosisstring_code_dict[s] = dx_list[0]
        else:
            have_to_find3.append(s)
            
    print('Number of dx we have to find...{}'.format(len(have_to_find3)))

    # hierarchy abovs
    dict_keys = list(diagnosisstring_code_dict.keys())
    have_to_find4 = []

    for i in range(len(have_to_find3)):
        s = "|".join(have_to_find3[i].split('|')[:-1])
        dx_list = []
        for k in dict_keys:
            if k[:len(s)] == s:
                dx_list.append(diagnosisstring_code_dict[k])
                
        dx_list = list(set(dx_list))
        if len(dx_list) == 1:
            diagnosisstring_code_dict[have_to_find3[i]] = dx_list[0]
        else:
            have_to_find4.append(have_to_find3[i])

    print('Number of dx we have to find...{}'.format(len(have_to_find4)))

    for t in range(4):
        c = -t-1
    dict_keys = list(diagnosisstring_code_dict.keys())
    have_to_find_l = []
    for i in range(len(have_to_find4)):
        s = have_to_find4[i]
        s = "|".join(s.split("|")[:c])
        dx_list =[]
        for k in dict_keys:
            if k[:len(s)] == s:
                dx_list.append(diagnosisstring_code_dict[k])
        dx_list2 = list(set(dx_list))
        if len(dx_list2) > 1:
            cnt = Counter(dx_list)
            mode = cnt.most_common(1)
            diagnosisstring_code_dict[have_to_find4[i]] = mode[0][0]
        else:
            have_to_find_l.append(have_to_find4[i])
    del(have_to_find4)
    have_to_find4 = have_to_find_l.copy()
    print('Number of dx we have to find...{}'.format(len(have_to_find4)))

    dx_depth1 = []
    dx_depth1_unique = []
    solution = lambda data: [x for x in set(data) if data.count(x) != 1]

    for ICD_list in eicu_dx_df['diagnosisstring']:
        single_list = list(pd.Series(ICD_list).map(diagnosisstring_code_dict))
        dx_depth1.append(single_list)
        dx_depth1_unique.append(list(set(single_list)))
    eicu_dx_df['diagnosis'] = pd.Series(dx_depth1_unique)

    return eicu_dx_df