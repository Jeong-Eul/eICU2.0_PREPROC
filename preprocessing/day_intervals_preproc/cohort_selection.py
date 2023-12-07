import pandas as pd
import numpy as np
import importlib
from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("/Users/DAHS/Desktop/early_prediction_of_circ_scl/eicu-crd/preprocessing_data/cohort"):
    os.makedirs("/Users/DAHS/Desktop/early_prediction_of_circ_scl/eicu-crd/preprocessing_data/cohort")

def get_visit_pts(eicu_path, group_col, visit_col, adm_visit_col, admit_col, disch_col):
    visit = pd.read_csv(eicu_path + "/patient.csv.gz", compression='gzip')
    
    visit[admit_col] = 0
    visit['los(min)'] = visit[disch_col]-visit[admit_col]

    visit['Age'] = visit['age'].str.extract('(\d+)').astype(float)
    visit['min_valid_year'] = visit['hospitaldischargeyear']
    
    visit['los(min)']=visit['los(min)'].astype(int)
    
    visit = visit.loc[visit['Age'] >= 18]
    
    return visit[[group_col, visit_col, adm_visit_col, admit_col, disch_col,'los(min)', 'min_valid_year', 'unitdischargestatus',
                  'Age','gender','ethnicity', 'admissionheight', 'admissionweight', 'dischargeweight']]
    
def partition_by_mort(df:pd.DataFrame, group_col:str, visit_col:str, admit_col:str, disch_col:str, death_col:str):
    """Applies labels to individual visits according to whether or not a death has occurred within
    the times of the specified admit_col and disch_col"""
    
    invalid = df.loc[(df[admit_col].isna()) | (df[disch_col].isna())]

    cohort = df.loc[(~df[admit_col].isna()) & (~df[disch_col].isna())]
    cohort['label']=0
    
    #cohort=cohort.fillna(0)
    pos_cohort=cohort[cohort[death_col] == 'Expired']
    neg_cohort=cohort[cohort[death_col] != 'Expired']
    neg_cohort=neg_cohort.fillna(0)
    pos_cohort=pos_cohort.fillna(0)
    
    pos_cohort['label'] = np.where((pos_cohort[disch_col] >= pos_cohort[admit_col]) & (pos_cohort[disch_col] <= pos_cohort[disch_col]),1,0)
    pos_cohort['label'] = pos_cohort['label'].astype("Int32")
    
    cohort=pd.concat([pos_cohort,neg_cohort], axis=0)
    cohort=cohort.sort_values(by=[group_col,admit_col])
    #print("cohort",cohort.shape)
    print("[ MORTALITY LABELS FINISHED ]")
    return cohort, invalid

def get_case_ctrls(df:pd.DataFrame, gap:int, group_col:str, visit_col:str, admit_col:str, disch_col:str, valid_col:str, death_col:str, use_mort=False,use_admn=False,use_los=False) -> pd.DataFrame:
    """Handles logic for creating the labelled cohort based on arguments passed to extract().

    Parameters:
    df: dataframe with patient data
    gap: specified time interval gap for readmissions
    group_col: patient identifier to group patients (normally subject_id)
    visit_col: visit identifier for individual patient visits (normally hadm_id or stay_id)
    admit_col: column for visit start date information (normally admittime or intime)
    disch_col: column for visit end date information (normally dischtime or outtime)
    valid_col: generated column containing a patient's year that corresponds to the 2017-2019 anchor time range
    dod_col: Date of death column
    """
    
    return partition_by_mort(df, group_col, visit_col, admit_col, disch_col, death_col)

def extract_data(root_dir):
    
    """Extracts cohort data and summary from MIMIC-IV data based on provided parameters.

    Parameters:
    cohort_output: name of labelled cohort output file
    summary_output: name of summary output file
    """
    print("===========eICU v2.0============")
    
    cohort_output="cohort_"
    summary_output="summary_"
    ICU='eICU'
    label= 'Mortality'
    
    group_col='uniquepid'
    visit_col='patientunitstayid'
    admit_col= 'unitadmissionoffset' # set '0' after
    disch_col='unitdischargeoffset' 
    death_col='unitdischargestatus' # dead if value == 'Expired'
    adm_visit_col='patienthealthsystemstayid'
    
    pts = get_visit_pts(
        eicu_path=root_dir+"/eicu-crd/2.0/",
        group_col=group_col,
        visit_col=visit_col,
        admit_col=admit_col,
        disch_col=disch_col,
        adm_visit_col=adm_visit_col,
    )
    
    cols = [group_col, visit_col, admit_col, disch_col, 'Age','gender','ethnicity','admissionheight', 'admissionweight','label']
    cols.append(death_col)

    cohort, invalid = get_case_ctrls(pts, None, group_col, visit_col, admit_col, disch_col,'min_valid_year', death_col, use_mort=True,use_admn=False,use_los=False)
    cols.append(adm_visit_col)
    
    cohort[cols].to_csv("/Users/DAHS/Desktop/early_prediction_of_circ_scl/eicu-crd/preprocessing_data/cohort/"+cohort_output+".csv.gz", index=False, compression='gzip')
    print("[ COHORT SUCCESSFULLY SAVED ]")
    
    summary = "\n".join([
        f"{label} FOR {ICU} DATA",
        f"# Admission Records: {cohort.shape[0]}",
        f"# Patients: {cohort[group_col].nunique()}",
        f"# Positive cases: {cohort[cohort['label']==1].shape[0]}",
        f"# Negative cases: {cohort[cohort['label']==0].shape[0]}"
    ])

    # save basic summary of data
    with open(f"/Users/DAHS/Desktop/early_prediction_of_circ_scl/eicu-crd/preprocessing_data/cohort/{summary_output}.txt", "w") as f:
        f.write(summary)

    print("[ SUMMARY SUCCESSFULLY SAVED ]")
    print(summary)

    return cohort_output