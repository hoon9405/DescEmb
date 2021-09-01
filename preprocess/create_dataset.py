import pandas as pd
import numpy as np
import os
import pickle
import warnings
import easydict

warnings.filterwarnings( 'ignore' )

# Create MIMIC-III dataset
def create_MIMIC_dataset(input_path, output_path):
    patient_path = os.path.join(input_path, 'PATIENTS.csv')
    icustay_path = os.path.join(input_path, 'eicu', 'ICUSTAYS.csv')
    dx_path = os.path.join(input_path, 'mimic', 'DIAGNOSES_ICD.csv')

    patients = pd.read_csv(patient_path)
    icus = pd.read_csv(icustay_path)
    dx = pd.read_csv(dx_path)
    print('length of PATIENTS.csv  : ', len(patients))
    print('length of ICUSTAYS.csv  : ', len(icus))
    print('length of DIAGNOSIS_ICD.csv  : ', len(icus))

    temp = icus[(icus['FIRST_CAREUNIT'] == icus['LAST_CAREUNIT'])]
    temp = temp[temp.LAST_CAREUNIT == 'MICU']
    
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
    cohort['los>3day'] = (cohort['LOS'] > 3.).astype('int64')
    cohort['los>7day'] = (cohort['LOS'] > 7.).astype('int64')
    cohort12h = cohort[cohort['LOS'] > 0.5]
    cohort12hDx = dx[dx.HADM_ID.isin(cohort12h.HADM_ID)]
    diagnosis = cohort12hDx.groupby('HADM_ID')['ICD9_CODE'].apply(list).to_frame()
    tempdf = cohort12h.join(diagnosis, on='HADM_ID')

    tempdf['12h_obs'] = tempdf.INTIME + pd.Timedelta(pd.offsets.Hour(12))
    tempdf['24h_obs'] = tempdf.INTIME + pd.Timedelta(pd.offsets.Hour(24))

    cohort_mm = tempdf.copy()
    pickle.dump(cohort_mm, open(os.path.join(output_path, 'mimic_cohort.pk'), 'wb'), -1)


# Create eICU dataset
def create_eICU_dataset(input_path, output_path):
    patient_path = os.path.join(input_path, 'patient.csv')
    patient_df = pd.read_csv(patient_path)
    dx_path = os.path.join(input_path, 'diagnosis.csv')
    dx = pd.read_csv(dx_path)

    print('Unique patient unit stayid : ', len(set(patient_df.patientunitstayid)))

    micu = patient_df[patient_df.unittype == 'MICU']

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
    cohort['los>3day'] = (cohort['losday'] > 3.).astype('int64')
    cohort['los>7day'] = (cohort['losday'] > 7.).astype('int64')

    cohort12h = cohort[cohort['unitdischargeoffset'] > 60*12]
    cohort12hDx = dx[dx.patientunitstayid.isin(cohort12h.patientunitstayid)]
    diagnosis = cohort12hDx.groupby('patientunitstayid')['diagnosisstring'].apply(list).to_frame()

    dxDict = dict(enumerate(cohort12hDx.groupby('diagnosisstring').count().index))
    dxDict = dict([(v,k) for k,v in dxDict.items()])
    print('diganosis unique : ', len(dxDict))

    tempdf = cohort12h.join(diagnosis, on='patientunitstayid')

    cohort_ei = tempdf.copy()
    pickle.dump(cohort_ei, open(os.path.join(input_path, 'eicu_cohort.pk'), 'wb'), -1)


def main():
    wd = os.getcwd()
    print('working directory .. : ', wd)

    args = easydict.EasyDict({'data_input_path' : '/home/ghhur/data/csv',
                            'data_output_path' : '/home/ghhur/data/csv'})

    mimic_path = os.path.join(args.data_input_path, 'mimic')
    eicu_path = os.path.join(args.data_input_path, 'eicu')

    create_MIMIC_dataset(mimic_path, mimic_path)
    create_eICU_dataset(eicu_path, eicu_path)

if __name__ == '__main__':
    main()