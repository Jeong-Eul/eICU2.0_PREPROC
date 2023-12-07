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

# import utils.outlier_removal
# from utils.outlier_removal import *  
# importlib.reload(utils.outlier_removal)
# import utils.outlier_removal
# from utils.outlier_removal import *

# import utils.uom_conversion
# from utils.uom_conversion import *  
# importlib.reload(utils.uom_conversion)
# import utils.uom_conversion
# from utils.uom_conversion import *

local = "/Users/DAHS/Desktop/early_prediction_of_circ_scl"+"/eicu-crd/preprocessing_data"

if not os.path.exists(local+"/features"):
    os.makedirs(local+"/features")
    
    
def feature_icu(root_dir, cohort_output, version_path, diag_flag=False,out_flag=True,chart_flag=True,proc_flag=True,med_flag=True, ing_flag=True, lab_flag=True):
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
        
    if lab_flag:
        print("[EXTRACTING LABS DATA]")
        lab = preproc_labs(root_dir+'/'+version_path+"/lab.csv.gz", root_dir+'/'+version_path+"/customLab.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        lab[['uniquepid','patientunitstayid', 'labid', 'labname', 'lab_time_from_admit','labresult']].to_csv(local+'/features/preproc_labs.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED LABS DATA]")
    
    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc = preproc_proc(root_dir+'/'+version_path+"/treatment.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz', 'treatmentoffset', dtypes=None, usecols=['patientunitstayid','treatmentoffset','treatmentid'])
        proc[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'treatmentid', 'treatmentoffset', 'unitadmissionoffset', 'event_time_from_admit']].to_csv(local+"/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    
    if med_flag:
        print("[EXTRACTING MEDICATIONS DATA]")
        med = preproc_meds(root_dir+'/'+version_path+"/intakeOutput.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        med[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'start_hours_from_admit', 'stop_hours_from_admit', 'celllabel', 'cellvaluenumeric']].to_csv(local+'/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
        
    if ing_flag:
        print("[EXTRACTING INGREDIENTS DATA]")
        ing = preproc_ings(root_dir+'/'+version_path+"/infusionDrug.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        ing[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'infusiondrugid', 'start_hours_from_admit', 'stop_hours_from_admit', 'drugname','drugrate']].to_csv(local+'/features/preproc_ing_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED INGREDIENTS DATA]")
        
        
def features_selection_icu(local, cohort_output, diag_flag, proc_flag, med_flag, ing_flag, out_flag, lab_flag, chart_flag, group_diag, 
                           group_med, group_ing, group_proc, group_out, group_chart, clean_labs):
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = pd.read_csv(local+'/'+"features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv(local+'/'+"summary/total_item_id.csv",header=0)
            diag=diag[diag['new_icd_code'].isin(features['new_icd_code'].unique())]
        
            print("Total number of rows",diag.shape[0])
            diag.to_csv(local+'/'+"features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if med_flag:       
        if group_med:   
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv(local+'/'+"features/preproc_med_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv(local+'/'+"summary/total_item_id.csv",header=0)
            med=med[med['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",med.shape[0])
            med.to_csv(local+'/'+'features/preproc_med_icu.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
            
    if ing_flag:       
        if group_ing:   
            print("[FEATURE SELECTION INGREDIENTS DATA]")
            ing = pd.read_csv(local+'/'+"features/preproc_ing_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv(local+'/'+"summary/total_item_id.csv",header=0)
            ing=ing[ing['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",med.shape[0])
            ing.to_csv(local+'/'+'features/preproc_ing_icu.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED INGREDIENTS DATA]")
    
    
    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv(local+'/'+"features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv(local+'/'+"summary/total_item_id.csv",header=0)
            proc=proc[proc['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",proc.shape[0])
            proc.to_csv(local+'/'+"features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        
        
    if out_flag:
        if group_out:            
            print("[FEATURE SELECTION OUTPUT EVENTS DATA]")
            out = pd.read_csv(local+'/'+"features/preproc_out_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv(local+'/'+"summary/total_item_id.csv",header=0)
            out=out[out['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",out.shape[0])
            out.to_csv(local+'/'+"features/preproc_out_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")
            
    if chart_flag:
        if group_chart:            
            print("[FEATURE SELECTION CHART EVENTS DATA]")
            
            chart=pd.read_csv(local+'/'+"features/preproc_chart_icu.csv.gz", compression='gzip',header=0, index_col=None)
            
            features=pd.read_csv(local+'/'+"summary/total_item_id.csv",header=0)
            chart=chart[chart['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",chart.shape[0])
            chart.to_csv(local+'/'+"features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
            
    if lab_flag:
        if clean_labs:            
            print("[FEATURE SELECTION LABS DATA]")
            chunksize = 10000000
            labs=pd.DataFrame()
            for chunk in tqdm(pd.read_csv(local+'/'+"features/preproc_labs.csv.gz", compression='gzip',header=0, index_col=None,chunksize=chunksize)):
                if labs.empty:
                    labs=chunk
                else:
                    labs=labs.append(chunk, ignore_index=True)
            features=pd.read_csv(local+'/'+"summary/total_item_id.csv",header=0)
            labs=labs[labs['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",labs.shape[0])
            labs.to_csv(local+'/'+"features/preproc_labs.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED LABS DATA]")