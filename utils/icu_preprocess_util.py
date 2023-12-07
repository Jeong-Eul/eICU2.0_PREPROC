import csv
import numpy as np
import pandas as pd
import sys, os
import re
import ast
import datetime as dt
from tqdm import tqdm
import math
# import labs_preprocess_util
# from labs_preprocess_util import *
# from sklearn.preprocessing import MultiLabelBinarizer

def preproc_ings(module_path:str, adm_cohort_path:str) -> pd.DataFrame:
    
    """
    drug rate = concentration * infusion rate -> 이에따라 사실 여기서의 drug rate는 amount임
    총 투여 시간(min) = drug amount / drug rate
    따라서 mimic은 drug rate와 total amount를 곱해서 그 시간에 준 양을 계산했다면 eicu에서는 시간 쪼개 놓고 drug rate 값이 amount 자리에 들어가면 됨
    """
    
    adm = pd.read_csv(adm_cohort_path,compression='gzip', low_memory=False, usecols=['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'unitadmissionoffset'])
    ing = pd.read_csv(module_path, compression='gzip', low_memory=False, usecols=['patientunitstayid','infusiondrugid', 'drugname', 'infusionoffset','infusionrate','drugamount', 'drugrate', 'volumeoffluid'])
    ing = ing.merge(adm, left_on = 'patientunitstayid', right_on = 'patientunitstayid', how = 'inner')
    
    ing['start_hours_from_admit'] = ing['infusionoffset']
    
    ing['unit_time'] = ing['drugname'].apply(lambda x: re.findall(r'\(([^)]+)\)', str(x))[-1] if re.findall(r'\(([^)]+)\)', str(x)) else None)
    
    # 유니크한 값 추출
    # unique_unit_time = ing['unit_time'].unique()
    
    # 'mcg/kg/min', 'mcg/min', 'ml/hr', 'mg/min', 'units/hr', 'mg/hr',
    #    'units/min', 'mcg/kg/hr', 'mcg/hr', None, 'mg/kg/min', 'Unknown',
    #    'Aviva', 'mg/kg/hr', 'nanograms/kg/min', 'units/kg/hr', 'Scale B',
    #    'Scale A', 'mL', 'Human', 'ml', 'ARTERIAL LINE', 'BNP',
    #    'units/kg/min'
    
    ing['stop_hours_from_admit'] = 0
    ing['drugamount']= ing['drugamount'].fillna(0) 
    ing['infusionrate']= ing['infusionrate'].fillna(0) 
    ing['volumeoffluid']= ing['volumeoffluid'].fillna(0) 
    
    idx = ing[ing['infusionrate']==0].index
    indx = ing[~(ing['infusionrate']==0)].index
    
    ing['stop_hours_from_admit'].loc[idx] = ing['start_hours_from_admit'].loc[idx]
    ing['stop_hours_from_admit'].loc[indx] =  ((ing['volumeoffluid'].loc[indx]/ing['infusionrate'].loc[indx])*60).apply(math.ceil) + ing['start_hours_from_admit'].loc[indx]
    
    #print(ing.isna().sum())
    # ing=ing.dropna()
    null_columns = ['unitadmissionoffset', 'start_hours_from_admit', 'stop_hours_from_admit', 'infusionoffset']
    ing = ing.dropna(subset=null_columns)
    
    non_null_columns = [col for col in ing.columns if col not in null_columns]
    ing[non_null_columns] = ing[non_null_columns].fillna(0)
    
    # ing[['drugamount','drugrate']]=ing[['drugamount','drugrate']].fillna(0)
    print("# of unique type of drug: ", ing.infusiondrugid.nunique())
    print("# Admissions:  ", ing.patientunitstayid.nunique())
    print("# Total rows",  ing.shape[0])
    
    return ing

def preproc_meds(module_path:str, adm_cohort_path:str) -> pd.DataFrame:
      
    adm = pd.read_csv(adm_cohort_path,compression='gzip', low_memory=False, usecols=['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'unitadmissionoffset'])
    med = pd.read_csv(module_path, compression='gzip', low_memory=False, usecols=['patientunitstayid','intakeoutputoffset','celllabel', 'cellvaluenumeric', 'cellpath'])
    med = med[med['cellpath'].str.contains('intake', case=False, regex=True)]
    
    med = med.merge(adm, left_on = 'patientunitstayid', right_on = 'patientunitstayid', how = 'inner')
    
    med['start_hours_from_admit'] = med['intakeoutputoffset']
    med['stop_hours_from_admit'] = med['intakeoutputoffset'] + 1
    
    #print(med.isna().sum())
    # med=med.dropna()
    null_columns = ['unitadmissionoffset', 'start_hours_from_admit', 'stop_hours_from_admit']
    med = med.dropna(subset=null_columns)
    non_null_columns = [col for col in med.columns if col not in null_columns]
    med[non_null_columns] = med[non_null_columns].fillna(0)
    
    # med[['amount','rate']]=med[['amount','rate']].fillna(0)
    # print("# of unique type of drug: ", med.itemid.nunique())
    print("# Admissions:  ", med.patientunitstayid.nunique())
    print("# Total rows",  med.shape[0])
    
    return med

def preproc_out(module_path:str, adm_cohort_path:str) -> pd.DataFrame:
      
    adm = pd.read_csv(adm_cohort_path,compression='gzip', low_memory=False, usecols=['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'unitadmissionoffset', 'unitdischargeoffset'])
    out = pd.read_csv(module_path, compression='gzip', low_memory=False, usecols=['patientunitstayid','intakeoutputoffset','celllabel', 'cellvaluenumeric', 'cellpath'])
    out = out[out['cellpath'].str.contains('output', case=False, regex=True)]
    
    out = out.merge(adm, left_on = 'patientunitstayid', right_on = 'patientunitstayid', how = 'inner')
    
    out['event_time_from_admit'] = out['intakeoutputoffset']
    
    null_columns = ['unitadmissionoffset', 'unitdischargeoffset', 'event_time_from_admit']
    out = out.dropna(subset=null_columns)
    non_null_columns = [col for col in out.columns if col not in null_columns]
    out[non_null_columns] = out[non_null_columns].fillna(0)
    
    print("# Admissions:  ", out.patientunitstayid.nunique())
    print("# Total rows",  out.shape[0])
    
    return out


def preproc_proc(dataset_path: str, cohort_path:str, time_col:str, dtypes: dict, usecols: list) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""

    def merge_module_cohort() -> pd.DataFrame:
        """Gets the initial module data with patients anchor year data and only the year of the charttime"""
        
        # read module w/ custom params
        module = pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes).drop_duplicates()
        #print(module.head())
        # Only consider values in our cohort
        cohort = pd.read_csv(cohort_path, compression='gzip', low_memory=False, usecols=['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'unitadmissionoffset', 'unitdischargeoffset'])
        
        #print(module.head())
        #print(cohort.head())

        # merge module and cohort
        return module.merge(cohort[['uniquepid','patienthealthsystemstayid','patientunitstayid', 'unitadmissionoffset','unitdischargeoffset']], how='inner', left_on='patientunitstayid', right_on='patientunitstayid')

    df_cohort = merge_module_cohort()
    df_cohort['event_time_from_admit'] = df_cohort[time_col]
    
    null_columns = ['unitadmissionoffset','unitdischargeoffset', 'event_time_from_admit']
    df_cohort = df_cohort.dropna(subset=null_columns)
    
    non_null_columns = [col for col in df_cohort.columns if col not in null_columns]
    df_cohort[non_null_columns] = df_cohort[non_null_columns].fillna(0)
    
    # Print unique counts and value_counts
    print("# Unique Events:  ", df_cohort.treatmentid.dropna().nunique())
    print("# Admissions:  ", df_cohort.patientunitstayid.nunique())
    print("Total rows", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort


def preproc_labs(lab_path: str, customlab_path: str, cohort_path:str) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""
    
    dataset_path_list = [lab_path, customlab_path]
    
    df_cohort=pd.DataFrame()
    cohort = pd.read_csv(cohort_path, low_memory=False, compression='gzip')
        
    for dataset_path in dataset_path_list:
        
        if dataset_path == lab_path:
            
            usecols = ['labid', 'patientunitstayid', 'labresultoffset', 'labresult', 'labname', 'labmeasurenamesystem']
            dtypes = {
                        'labid':'int64',
                        'patientunitstayid':'int64',
                        'labresultoffset':'int64',      
                        'labresult':'object',
                        'labname':'object',
                        'labmeasurenamesystem':'object',
                        }
        elif dataset_path == customlab_path:
            
            usecols = ['customlabid', 'patientunitstayid', 'labotheroffset', 'labotherresult', 'labothername']
            dtypes = {
                        'customlabid':'int64',
                        'patientunitstayid':'int64',
                        'labotheroffset':'int64',      
                        'labotherresult':'object',
                        'labothername':'object',
                        }

        chunksize = 10000000
        for chunk in tqdm(pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes, chunksize=chunksize)):

            if dataset_path == lab_path:
                chunk=chunk.dropna(subset=['labresult'])
                chunk['labmeasurenamesystem']=chunk['labmeasurenamesystem'].fillna(0)
            else:
                chunk=chunk.dropna(subset=['labotherresult'])
                chunk['labmeasurenamesystem']='Not exist'
            
            chunk=chunk[chunk['patientunitstayid'].isin(cohort['patientunitstayid'].unique())]

            if dataset_path == customlab_path:
                chunk=chunk.rename(columns={'customlabid':'labid', 'labotheroffset':'labresultoffset', 'labotherresult':'labresult',
                                            'labothername':'labname'})

            chunk = chunk.merge(cohort[['uniquepid','patientunitstayid', 'unitadmissionoffset', 'unitdischargeoffset']], how='inner', left_on='patientunitstayid', right_on='patientunitstayid')
    
            chunk['lab_time_from_admit'] = chunk['labresultoffset']
            chunk=chunk.dropna()
            
            if df_cohort.empty:
                df_cohort=chunk
            else:
                df_cohort=df_cohort.append(chunk, ignore_index=True)

    print("# Itemid: ", df_cohort.labid.nunique())
    print("# Admissions: ", df_cohort.patientunitstayid.nunique())
    print("Total number of rows: ", df_cohort.shape[0])
    
    return df_cohort


def preproc_chart(chart_path: str, respir_path: str, cohort_path:str) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""
    
    # Only consider values in our cohort
    cohort = pd.read_csv(cohort_path, low_memory=False, compression='gzip')
    df_cohort=pd.DataFrame()
    
    chunksize = 10000000
    
    dataset_path_list = [chart_path, respir_path]
    for dataset_path in dataset_path_list:
        
        if dataset_path == chart_path:
            
            usecols = ['nursingchartid', 'patientunitstayid', 'nursingchartoffset', 'nursingchartvalue', 'nursingchartcelltypevallabel']
            dtypes = {
                        'nursingchartid':'int64',
                        'patientunitstayid':'int64',
                        'nursingchartoffset':'int64',      
                        'nursingchartvalue':'object',
                        'nursingchartcelltypevallabel':'object'
                        }
            
        elif dataset_path == respir_path:
            
            usecols = ['respchartid', 'patientunitstayid', 'respchartoffset', 'respchartvaluelabel', 'respchartvalue']
            dtypes = {
                        'respchartid':'int64',
                        'patientunitstayid':'int64',
                        'respchartoffset':'int64',      
                        'respchartvalue':'object',
                        'respchartvaluelabel':'object',
                        }
    
        for chunk in tqdm(pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes, chunksize=chunksize)):
            
            if dataset_path == chart_path:
                chunk=chunk.dropna(subset=['nursingchartvalue'])
                chunk['event_time_from_admit'] = chunk['nursingchartoffset']
                
            elif dataset_path == respir_path:
                
                chunk=chunk.dropna(subset=['respchartvalue'])
                chunk['event_time_from_admit'] = chunk['respchartoffset']
                
                chunk=chunk.rename(columns={'respchartid':'nursingchartid', 'respchartoffset':'nursingchartoffset', 'respchartvalue':'nursingchartvalue',
                                            'respchartvaluelabel':'nursingchartcelltypevallabel'})

            chunk_merged=chunk.merge(cohort[['patientunitstayid', 'uniquepid']], how='inner', left_on='patientunitstayid', right_on='patientunitstayid')
            null_columns = ['nursingchartvalue']
            non_null_columns = [col for col in df_cohort.columns if col not in null_columns]
            
            chunk_merged=chunk_merged.dropna(subset = non_null_columns)
            chunk_merged=chunk_merged.drop_duplicates()
            if df_cohort.empty:
                df_cohort=chunk_merged
            else:
                df_cohort=df_cohort.append(chunk_merged, ignore_index=True)
        
        

    print("# Unique Events:  ", df_cohort.nursingchartid.nunique())
    print("# Admissions:  ", df_cohort.patientunitstayid.nunique())
    print("Total rows", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort
