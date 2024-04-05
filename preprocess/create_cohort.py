import pandas as pd
import numpy as np
import os
import pickle
import warnings
import shutil
import easydict
from collections import Counter
import subprocess

warnings.filterwarnings( 'ignore' )

def create_cohort(src_data, dataset_path, dest_path, ccs_dx_tool_path, icd10to9_path, min_stay_hours):
    function_mapping = {
        'mimiciii': create_MIMIC_cohort,
        'eicu': create_eICU_cohort
    }
    
    if src_data in function_mapping:
        function_mapping[src_data](dataset_path, dest_path, ccs_dx_tool_path, icd10to9_path, min_stay_hours)
    else:
        print(f"Unsupported data source: {src_data}")


def create_MIMIC_cohort(dataset_path, dest_path, ccs_dx_path, icd10to9_path, min_stay_hours):
    # Check MIMIC-III dataset
    if not os.path.exists(dataset_path):
        print("Data is not found so try to download from the internet. "
            "Note that this is a restricted-access resource. "
            "Please log in to physionet.org with a credentialed user.")
        download_ehr_from_url(
            url="https://physionet.org/files/mimiciv/2.0/", dest=dataset_path
        )
    print("Data directory is set to {}".format(dataset_path))
    
    if os.path.exists(os.path.join(dest_path, 'mimiciii_cohort.pkl')):
        print('mimic_cohort.pkl already exists skip create cohort step!___')
        return 
    
    # Load MIMIC-III dataset   
    patient_path = os.path.join(dataset_path, 'PATIENTS.csv')
    icustay_path = os.path.join(dataset_path, 'ICUSTAYS.csv')
    dx_path = os.path.join(dataset_path, 'DIAGNOSES_ICD.csv')

    patients = pd.read_csv(patient_path)
    icus = pd.read_csv(icustay_path)
    dx = pd.read_csv(dx_path)
    print('length of PATIENTS.csv  : ', len(patients))
    print('length of ICUSTAYS.csv  : ', len(icus))
    print('length of DIAGNOSIS_ICD.csv  : ', len(icus))

    # Create MIMIC cohort
    icus = icus.drop(columns=['ROW_ID'])
    icus = icus[(icus['FIRST_CAREUNIT'] == icus['LAST_CAREUNIT'])]
    icus = icus[icus.LAST_CAREUNIT == 'MICU']     
    
    # Convert to datetime
    icus['INTIME'] = pd.to_datetime(icus['INTIME'], infer_datetime_format=True)
    icus['OUTTIME'] = pd.to_datetime(icus['OUTTIME'], infer_datetime_format=True)

    patients['DOB'] = pd.to_datetime(patients['DOB'], infer_datetime_format=True)
    patients['DOD'] = pd.to_datetime(patients['DOD'], infer_datetime_format=True)
    patients['DOD_HOSP'] = pd.to_datetime(patients['DOD_HOSP'], infer_datetime_format=True)
    patients['DOD_SSN'] = pd.to_datetime(patients['DOD_SSN'], infer_datetime_format=True)
    patients = patients.drop(columns=['ROW_ID'])

    # Filter patients
    small_patients = patients[patients.SUBJECT_ID.isin(icus.SUBJECT_ID)]
    icus = icus.merge(small_patients, on='SUBJECT_ID', how='left')

    # Calculate age
    datediff = np.array(icus.INTIME.dt.date) - np.array(icus.DOB.dt.date)
    age = np.array([x.days // 365 for x in datediff])
    icus['age'] = age
    
    # Filter patients
    icus = icus[icus.age >= 18]
    print('length of icus  :', len(icus))

    # Create readmission label
    readmit = icus.groupby('HADM_ID')['ICUSTAY_ID'].count()
    readmit_labels = (readmit > 1).astype('int64').to_frame().rename(
        columns={'ICUSTAY_ID':'readmission'}
        )
    print('readmission value counts :', readmit_labels.value_counts())

    # filter only first icu
    small_icus = icus.loc[icus.groupby('HADM_ID').INTIME.idxmin()]
    cohort = small_icus.join(readmit_labels, on='HADM_ID')
    
    # Create mortality label
    dead = cohort[~pd.isnull(cohort.DOD_HOSP)].copy()
    dead_labels = (
        (dead.DOD_HOSP.dt.date > dead.INTIME.dt.date) & 
        (dead.DOD_HOSP.dt.date <= dead.OUTTIME.dt.date)
        )
    dead_labels = dead_labels.astype('int64')
    dead['mortality'] = np.array(dead_labels)
    cohort = cohort.merge(dead.iloc[:, [2,-1]], on='ICUSTAY_ID', how='left')
    cohort['mortality'] = cohort['mortality'].fillna(0)
    cohort = cohort.astype({'mortality': int})
    
    # Create Lenth of stay label
    cohort['los_3day'] = (cohort['LOS'] > 3.).astype('int64')
    cohort['los_7day'] = (cohort['LOS'] > 7.).astype('int64')
    cohort12h = cohort[cohort['LOS'] > min_stay_hours/24]
    cohort12hDx = dx[dx.HADM_ID.isin(cohort12h.HADM_ID)]
    
    # Create diagnosis label
    diagnosis = cohort12hDx.groupby('HADM_ID')['ICD9_CODE'].apply(list).to_frame()
    tempdf = cohort12h.join(diagnosis, on='HADM_ID')

    # Mark 12h and 24h observation time
    tempdf['12h_obs'] = tempdf.INTIME + pd.Timedelta(pd.offsets.Hour(12))
    tempdf['24h_obs'] = tempdf.INTIME + pd.Timedelta(pd.offsets.Hour(24))

    cohort = tempdf.reset_index(drop=True).copy()
    cohort = mimic_diagnosis_label(cohort, dest_path, ccs_dx_path)    

    #save as pickle
    pickle.dump(cohort, open(os.path.join(dest_path, 'mimiciii_cohort.pkl'), 'wb'), -1)


# Create eICU dataset
def create_eICU_cohort(dataset_path, dest_path, ccs_dx_path, icd10to9_path, min_stay_hours):
    # Check eICU dataset
    if not os.path.exists(dataset_path):
        print("Data is not found so try to download from the internet. "
            "Note that this is a restricted-access resource. "
            "Please log in to physionet.org with a credentialed user.")
        download_ehr_from_url(
            url="https://physionet.org/files/eicu-crd/2.0/", dest=dataset_path
        )
    print("Data directory is set to {}".format(dataset_path))
    
    if os.path.exists(os.path.join(dest_path, 'eicu_cohort.pkl')):
        print('eicu_cohort.pkl already exists skip create cohort step!___')
        return        
    
    # Load eICU dataset
    patient = pd.read_csv(os.path.join(dataset_path, 'patient.csv'))
    dx = pd.read_csv(os.path.join(dataset_path, 'diagnosis.csv'))

    print('Unique patient unit stayid : ', len(set(patient.patientunitstayid)))

    micu = patient[patient.unittype == 'MICU']  

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

    # Create mortality label
    cohort['mortality'] = (cohort['unitdischargestatus'] == 'Expired').astype('int64')
    cohort['losday'] = (cohort['unitdischargeoffset'].astype('float') / (24.*60.))
    cohort['los_3day'] = (cohort['losday'] > 3.).astype('int64')
    cohort['los_7day'] = (cohort['losday'] > 7.).astype('int64')

    # filter by min_LOS
    cohort12h = cohort[cohort['unitdischargeoffset'] > 60*min_stay_hours]
    cohort12hDx = dx[dx.patientunitstayid.isin(cohort12h.patientunitstayid)]
    diagnosis = cohort12hDx.groupby('patientunitstayid')['diagnosisstring'].apply(list).to_frame()

    dxDict = dict(enumerate(cohort12hDx.groupby('diagnosisstring').count().index))
    dxDict = dict([(v,k) for k,v in dxDict.items()])
    print('diganosis unique : ', len(dxDict))

    tempdf = cohort12h.join(diagnosis, on='patientunitstayid') 
    cohort = tempdf.copy().reset_index(drop=True)

    cohort = eicu_diagnosis_label(cohort, dest_path, dx, ccs_dx_path, icd10to9_path)

    cohort = cohort[cohort['diagnosis'] !=float].reset_index(drop=True)
    cohort = cohort.reset_index(drop=True)
    pickle.dump(cohort, open(os.path.join(dest_path, 'eicu_cohort.pkl'), 'wb'), -1)


def mimic_diagnosis_label(cohort, dest_path, ccs_dx_path=None):
    #diagnosis label
    if ccs_dx_path is None:
        ccs_dx_path = os.path.join(dest_path, "ccs_multi_dx_tool_2015.csv")

        if not os.path.exists(ccs_dx_path):
            print(
                "`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet."
            )
            download_ccs_from_url(dest_path)
        
    ccs_dx = pd.read_csv(ccs_dx_path)
    level1_dict = load_and_prepare_ccs_dx(ccs_dx)
    
    if not os.path.exists(os.path.join(dest_path, 'dx_label_mapping.pkl')):
        print('dx_label_mapping.pkl dx mapping pkl save___')
        ccs_dx_mapping_save(ccs_dx, dest_path)
    
    dx1_list = []
    for idx, dxx in enumerate(cohort['ICD9_CODE']):
        if pd.isna(dxx).all():  
            dx1_list.append([]) 
            continue

        one_list = []
        for dx in dxx:
            if pd.isna(dx): 
                continue 
            dx1 = level1_dict.get(dx)
            if dx1:
                one_list.append(dx1)
        dx1_list.append(list(set(one_list)))


    cohort['diagnosis'] = pd.Series(dx1_list)
    cohort = cohort[cohort['diagnosis'] !=float].reset_index(drop=True)
    dx1_length = [len(i) for i in dx1_list]
    print("average length: ", np.array(dx1_length).mean())
    print('dx freqeuncy', np.bincount(dx1_length))
    print("max length: ", np.array(dx1_length).max())
    print("min length: ", np.array(dx1_length).min())

    return cohort

def eicu_diagnosis_label(cohort, dest_path, eicu_dx, ccs_dx_path=None, icd10to9_path=None):
    if ccs_dx_path is None:
        ccs_dx_path = os.path.join(dest_path, "ccs_multi_dx_tool_2015.csv")

        if not os.path.exists(ccs_dx_path):
            print(
                "`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet."
            )
            download_ccs_from_url(dest_path)
        
    if icd10to9_path is None:
        icd10to9_path = os.path.join(dest_path, "icd10cmtoicd9gem.csv")

        if not os.path.exists(icd10to9_path):
            print(
                "`icd10cmtoicd9gem.csv` is not found so try to download from the internet."
            )
            download_icdgem_from_url(dest_path)    
    
    icd10_icd9 = pd.read_csv(icd10to9_path)
    ccs_dx = pd.read_csv(ccs_dx_path)
    
    if not os.path.exists(os.path.join(dest_path, 'dx_label_mapping.pkl')):
        print('dx_label_mapping.pkl dx mapping pkl save___')
        ccs_dx_mapping_save(ccs_dx, dest_path)
    
    level1_dict = load_and_prepare_ccs_dx(ccs_dx)

    cohort = cohort.dropna(subset=['diagnosisstring']).reset_index(drop=True)
    dx_unique = set(x for dx_list in cohort['diagnosisstring'] for x in dx_list)

    eicu_dx = eicu_dx.dropna(subset=['icd9code'])
    eicu_dx['icd9code_clean'] = eicu_dx['icd9code'].apply(lambda x: x.replace('.', '').strip())

    # drop the icd9code NaN for right now
    eicu_dx = eicu_dx.dropna(subset=['icd9code']).copy().reset_index(drop=True)

    # make diagnosisstring - ICD9 code dictionary
    diagnosisstring_code_dict = {}
    key_error_list = []

    for diagnosis_string, icd9code in zip(eicu_dx['diagnosisstring'], eicu_dx['icd9code_clean']):
        icd9code_first = icd9code.split(',')[0] 
        ccs_level = level1_dict.get(icd9code_first)
        if ccs_level:
            diagnosisstring_code_dict[diagnosis_string] = ccs_level
        else:
            key_error_list.append(diagnosis_string)

    key_error_list = list(set(key_error_list))
    print(f'Number of diagnosis with only ICD 10 code: {len(key_error_list)}')

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
                diagnosisstring_code_dict[key_error_list[i]] = level1_dict[icd9code]
            except KeyError:
                icd10_key_error_list.append(key_error_list[i])
                
    print('Number of more than one icd10 codes : {}'.format(len(two_icd10_code_list)))
    print('Number of icd10key_error_list : {}'.format(len(icd10_key_error_list)))

    
    def _map_dx_to_ccs(icd_list, diagnosisstring_code_dict, class_list):
        for icd_code, ccs_class in zip(icd_list, class_list):
            
            diagnosisstring_code_dict[icd_code] = ccs_class

    def _fill_in_the_blanks(dx_unique, diagnosisstring_code_dict):
        have_to_find = set(dx_unique) - set(diagnosisstring_code_dict.keys())
        print(f'Number of dx we have to find: {len(have_to_find)}')
        return have_to_find

    class_list = ['6', '7', '6', '7', '2', '6', '6', '7', '6', '6','6']
    _map_dx_to_ccs(two_icd10_code_list, diagnosisstring_code_dict, class_list)

    missing_dx = _fill_in_the_blanks(dx_unique, diagnosisstring_code_dict)

    dict_keys = list(diagnosisstring_code_dict.keys())
    have_to_find_final = []
    for diagnosis in missing_dx:
        parts = diagnosis.split('|')
        for depth in range(len(parts), 0, -1):
            prefix = "|".join(parts[:depth])
            dx_list = [diagnosisstring_code_dict[k] for k in dict_keys if k.startswith(prefix)]
            if dx_list:
                dx_list_counted = Counter(dx_list)
                most_common_code, _ = dx_list_counted.most_common(1)[0]
                diagnosisstring_code_dict[diagnosis] = most_common_code
                break
        else: 
            have_to_find_final.append(diagnosis)

    if not have_to_find_final:
        print("all codes are mapped")
    else:
        print(f"number of remained codes: {len(have_to_find_final)}")
        
        for diagnosis in have_to_find_final:
            diagnosisstring_code_dict[diagnosis] = '18'


    cohort['diagnosis'] = cohort['diagnosisstring'].apply(
        lambda x: [diagnosisstring_code_dict.get(dx, 'Unknown') for dx in x]
        )
    cohort['diagnosis'] = cohort['diagnosis'].apply(
        lambda dx_list: list(set(dx_list)) if dx_list else []
        )
    cohort = cohort[cohort['diagnosis'].apply(len) > 0].reset_index(drop=True)
    
    return cohort


def ccs_dx_mapping_save(ccs_dx, dest_path):
    dx_label_dict = {}
    for label_num in ccs_dx["'CCS LVL 1'"].unique():
        ccs_dx_label = ccs_dx[ccs_dx["'CCS LVL 1'"] == label_num]["'CCS LVL 1 LABEL'"].values[0]
        dx_label_dict[label_num.strip("'")] = ccs_dx_label
    
    with open(os.path.join(dest_path, 'dx_label_mapping.pkl'), 'wb') as f:
        pickle.dump(dx_label_dict, f)


def load_and_prepare_ccs_dx(ccs_dx):
    ccs_dx = ccs_dx.rename(columns=lambda x: x.strip("'").strip())
    
    ccs_dx["ICD-9-CM CODE"] = ccs_dx["ICD-9-CM CODE"].str.strip("'").str.replace(" ", "")
    ccs_dx["CCS LVL 1"] = ccs_dx["CCS LVL 1"].str.strip("'")
    
    level1_dict = ccs_dx.set_index("ICD-9-CM CODE")["CCS LVL 1"].to_dict()
    return level1_dict


def download_ehr_from_url(url, dest) -> None:
        username = input("Email or Username: ")
        subprocess.run(
            [
                "wget",
                "-r",
                "-N",
                "-c",
                "np",
                "--user",
                username,
                "--ask-password",
                url,
                "-P",
                dest,
            ]
        )
        output_dir = url.replace("https://", "").replace("http://", "")

        if not os.path.exists(os.path.join(dest, output_dir)):
            raise AssertionError(
                "Download failed. Please check your network connection or "
                "if you log in with a credentialed user"
            )


def download_ccs_from_url(dest) -> None:
        subprocess.run(
            [
                "wget",
                "-N",
                "-c",
                "https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip",
                "-P",
                dest,
            ]
        )

        import zipfile

        with zipfile.ZipFile(
            os.path.join(dest, "Multi_Level_CCS_2015.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(os.path.join(dest, "foo.d"))
        os.rename(
            os.path.join(dest, "foo.d", "ccs_multi_dx_tool_2015.csv"),
            os.path.join(dest, "ccs_multi_dx_tool_2015.csv"),
        )
        os.remove(os.path.join(dest, "Multi_Level_CCS_2015.zip"))
        shutil.rmtree(os.path.join(dest, "foo.d"))


def download_icdgem_from_url(dest) -> None:
    subprocess.run(
        [
            "wget",
            "-N",
            "-c",
            "https://data.nber.org/gem/icd10cmtoicd9gem.csv",
            "-P",
            dest,
        ]
    )