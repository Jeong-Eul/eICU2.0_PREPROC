import os
import pickle
import glob
import importlib
#print(os.getcwd())
#os.chdir('../../')
#print(os.getcwd())
import utils.icu_preprocess_util
from utils.icu_preprocess_util import * 
importlib.reload(utils.icu_preprocess_util)
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *# module of preprocessing functions

import utils.outlier_removal
from utils.outlier_removal import *  
importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

# import utils.uom_conversion
# from utils.uom_conversion import *  
# importlib.reload(utils.uom_conversion)
# import utils.uom_conversion
# from utils.uom_conversion import *

local = "/Users/DAHS/Desktop/early_prediction_of_circ_scl"+"/eicu-crd/preprocessing_data"

if not os.path.exists(local+"/features"):
    os.makedirs(local+"/features")
    
    
def feature_icu(root_dir, cohort_output, version_path, diag_flag=False,out_flag=True,chart_flag=True,proc_flag=True,med_flag=True, ing_flag=True, lab_flag=True, pe_flag=True, microlab_flag=True, vent_flag=True):
    # if diag_flag:
    #     print("[EXTRACTING DIAGNOSIS DATA]")
    #     diag = preproc_icd_module(root_dir+version_path+"/hosp/diagnoses_icd.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz', './utils/mappings/ICD9_to_ICD10_mapping.txt', map_code_colname='diagnosis_code')
    #     diag[['subject_id', 'hadm_id', 'stay_id', 'icd_code','root_icd10_convert','root']].to_csv(local+"/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
    #     print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if out_flag:  
        print("[EXTRACTING OUPTPUT EVENTS DATA]")
        out = preproc_out(root_dir+'/'+version_path+"/intakeOutput.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        out[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'celllabel', 'unitadmissionoffset', 'event_time_from_admit', 'cellvaluenumeric']].to_csv(local+"/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED OUPTPUT EVENTS DATA]")
    
    if chart_flag:
        print("[EXTRACTING CHART EVENTS DATA]")
        chart=preproc_chart(root_dir+'/'+version_path+"/nurseCharting.csv.gz", root_dir+'/'+version_path+"/respiratoryCharting.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        chart[['uniquepid', 'patientunitstayid', 'nursingchartid','event_time_from_admit','nursingchartcelltypevallabel', 'nursingchartvalue']].to_csv(local+"/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
        
    if pe_flag:
        print("[EXTRACTING (A)PERIOD EVENTS DATA]")
        period=preproc_period(root_dir+'/'+version_path+"/vitalPeriodic.csv.gz", root_dir+'/'+version_path+"/vitalAperiodic.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        period[['uniquepid', 'patientunitstayid', 'nursingchartid','event_time_from_admit','nursingchartcelltypevallabel', 'nursingchartvalue']].to_csv(local+"/features/preproc_period_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED (A)PERIOD EVENTS DATA]")
        
    if lab_flag:
        print("[EXTRACTING LABS DATA]")
        lab = preproc_labs(root_dir+'/'+version_path+"/lab.csv.gz", root_dir+'/'+version_path+"/customLab.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        lab[['uniquepid','patientunitstayid', 'labid', 'labname', 'lab_time_from_admit','labresult']].to_csv(local+'/features/preproc_labs.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED LABS DATA]")
        
    if microlab_flag:
        print("[EXTRACTING LABS DATA]")
        microlab = preproc_microlabs(root_dir+'/'+version_path+"/microLab.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        microlab[['uniquepid','patientunitstayid', 'labid', 'labname', 'lab_time_from_admit']].to_csv(local+'/features/preproc_microlabs.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED LABS DATA]")
    
    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc = preproc_proc(root_dir+'/'+version_path+"/treatment.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz', 'treatmentoffset', dtypes=None, usecols=['patientunitstayid','treatmentoffset','treatmentid', 'treatmentstring'])
        proc[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'treatmentid', 'treatmentoffset','treatmentstring','unitadmissionoffset', 'event_time_from_admit']].to_csv(local+"/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        
    if med_flag:
        print("[EXTRACTING MEDICATIONS DATA]")
        med = preproc_meds(root_dir+'/'+version_path+"/intakeOutput.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        med[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'start_hours_from_admit', 'stop_hours_from_admit', 'celllabel', 'cellvaluenumeric']].to_csv(local+'/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
        
    if ing_flag:
        print("[EXTRACTING INGREDIENTS DATA]")
        ing = preproc_ings(root_dir+'/'+version_path+"/infusionDrug.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        ing[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'infusiondrugid', 'start_hours_from_admit', 'stop_hours_from_admit', 'drugname','drugrate', 'infusionrate']].to_csv(local+'/features/preproc_ing_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED INGREDIENTS DATA]")
        
    if vent_flag:
        print("[EXTRACTING VENTILATION DATA]")
        ing = preproc_vent(root_dir+'/'+version_path+"/respiratoryCharting.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        ing[['respchartid', 'patientunitstayid', 'ventstartoffset', 'ventendoffset', 'label']].to_csv(local+'/features/preproc_vent_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED VENTILATION DATA]")
        
        
def features_selection_icu(local, cohort_output,  diag_flag, proc_flag, med_flag, ing_flag, out_flag, lab_flag, chart_flag, microlab_flag,
                           group_diag, group_med, group_ing, group_proc, group_out, group_chart,clean_labs,  group_microlab):
    # if diag_flag:
    #     if group_diag:
    #         print("[FEATURE SELECTION DIAGNOSIS DATA]")
    #         diag = pd.read_csv(local+'/'+"features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
    #         features=pd.read_csv(local+'/'+"summary/total_item_id.csv",header=0)
    #         diag=diag[diag['new_icd_code'].isin(features['new_icd_code'].unique())]
        
    #         print("Total number of rows",diag.shape[0])
    #         diag.to_csv(local+'/'+"features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
    #         print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if med_flag:       
        if group_med:   
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv(local+'/'+"features/preproc_med_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv(local+'/'+"summary/total_med_infusion_intake.csv",header=0)
            med=med[med['celllabel'].isin(features['itemstring'].unique())]
            print("Total number of rows",med.shape[0])
            rename_dict = dict(zip(features.itemstring, features.altername))
            med['celllabel'] = med['celllabel'].map(rename_dict)
            med.to_csv(local+'/'+'features/preproc_med(selected)_icu.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
            
    if ing_flag:       
        if group_ing:   
            print("[FEATURE SELECTION INGREDIENTS DATA]")
            ing = pd.read_csv(local+'/'+"features/preproc_ing_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv(local+'/'+"summary/total_med_infusion_intake.csv",header=0)
            ing=ing[ing['drugname'].isin(features['itemstring'].unique())]
            print("Total number of rows",ing.shape[0])
            rename_dict = dict(zip(features.itemstring, features.altername))
            ing['drugname'] = ing['drugname'].map(rename_dict)
            ing.to_csv(local+'/'+'features/preproc_ing(selected)_icu.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED INGREDIENTS DATA]")
    
    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv(local+'/'+"features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv(local+'/'+"summary/total_procedure.csv",header=0)
            proc=proc[proc['treatmentstring'].isin(features['itemstring'].unique())]
            print("Total number of rows",proc.shape[0])
            rename_dict = dict(zip(features.itemstring, features.altername))
            proc['treatmentstring'] = proc['treatmentstring'].map(rename_dict)
            proc.to_csv(local+'/'+"features/preproc_proc(selected)_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        
    if out_flag:
        if group_out:            
            print("[FEATURE SELECTION OUTPUT EVENTS DATA]")
            out = pd.read_csv(local+'/'+"features/preproc_out_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv(local+'/'+"summary/total_output.csv",header=0)
            out=out[out['celllabel'].isin(features['itemstring'].unique())]
            print("Total number of rows",out.shape[0])
            rename_dict = dict(zip(features.itemstring, features.altername))
            out['celllabel'] = out['celllabel'].map(rename_dict)
            out.to_csv(local+'/'+"features/preproc_out(selected)_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")
            
    if chart_flag:
        if group_chart:            
            print("[FEATURE SELECTION CHART EVENTS DATA]")
            
            chart=pd.read_csv(local+'/'+"features/preproc_chart_icu.csv.gz", compression='gzip',header=0, index_col=None)
            period = pd.read_csv(local+'/'+"features/preproc_period_icu.csv.gz", compression='gzip',header=0, index_col=None)
            
            total_chart = pd.concat([chart, period], ignore_index=True)
            
            features=pd.read_csv(local+'/'+"summary/total_lab_chart.csv",header=0)
            total_chart=total_chart[total_chart['nursingchartcelltypevallabel'].isin(features['itemstring'].unique())]
            print("Total number of rows",total_chart.shape[0])
            rename_dict = dict(zip(features.itemstring, features.altername))
            total_chart['nursingchartcelltypevallabel'] = total_chart['nursingchartcelltypevallabel'].map(rename_dict)
            total_chart.to_csv(local+'/'+"features/preproc_chart(selected)_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
            
    if lab_flag:
        if clean_labs:            
            print("[FEATURE SELECTION LABS DATA]")
            labs=pd.read_csv(local+'/'+"features/preproc_labs.csv.gz", compression='gzip',header=0, index_col=None)

            features=pd.read_csv(local+'/'+"summary/total_lab_chart.csv",header=0)
            labs=labs[labs['labname'].isin(features['itemstring'].unique())]
            print("Total number of rows",labs.shape[0])
            rename_dict = dict(zip(features.itemstring, features.altername))
            labs['labname'] = labs['labname'].map(rename_dict)
            labs.to_csv(local+'/'+"features/preproc_labs(selected).csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED LABS DATA]")
        
    if microlab_flag:
        if group_microlab:            
            print("[FEATURE SELECTION MICROLABS DATA]")
            microlabs=pd.read_csv(local+'/'+"features/preproc_microlabs.csv.gz", compression='gzip',header=0, index_col=None)

            features=pd.read_csv(local+'/'+"summary/total_lab_chart.csv",header=0)
            microlabs=microlabs[microlabs['labname'].isin(features['itemstring'].unique())]
            print("Total number of rows",microlabs.shape[0])
            rename_dict = dict(zip(features.itemstring, features.altername))
            microlabs['labname'] = microlabs['labname'].map(rename_dict)
            microlabs.to_csv(local+'/'+"features/preproc_microlabs(selected).csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MICROLABS DATA]")
            
            
def preprocess_feature_icu(chart_flag,clean_chart,impute_outlier_chart,thresh,left_thresh,
                            lab_flag, imput_outlier_lab, thresh_lab, left_thresh_lab, clean_labs):
    
    if chart_flag:
        if clean_chart:
            print("[PROCESSING CHART EVENTS DATA]")
            chart = pd.read_csv(local+"/features/preproc_chart(selected)_icu.csv.gz", compression='gzip',header=0, low_memory=False)
            chart['nursingchartvalue'] = pd.to_numeric(chart['nursingchartvalue'], errors='coerce')
            chart = chart.dropna(subset=['nursingchartvalue'])
            chart = chart.reset_index(drop=True)
            chart = outlier_imputation(chart, 'nursingchartcelltypevallabel', 'nursingchartvalue', thresh,left_thresh,impute_outlier_chart)
            
            print("Total number of rows",chart.shape[0])
            chart.to_csv(local+"/features/preproc_chart(selected)_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
            
    if lab_flag:  
        if clean_labs:   
            print("[PROCESSING LABS DATA]")
            labs = pd.read_csv(local+"/features/preproc_labs(selected).csv.gz", compression='gzip',header=0, low_memory=False)
            labs = outlier_imputation(labs, 'labname', 'labresult', thresh_lab,left_thresh_lab,imput_outlier_lab)
            
            print("Total number of rows",labs.shape[0])
            labs.to_csv(local+"/features/preproc_labs(selected).csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED LABS DATA]")    