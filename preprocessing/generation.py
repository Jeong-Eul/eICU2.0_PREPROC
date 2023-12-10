import os 
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from pathlib import Path
import os
import importlib
import warnings

pd.set_option('mode.chained_assignment',  None) 
warnings.simplefilter(action='ignore', category=FutureWarning) 

local = "/Users/DAHS/Desktop/early_prediction_of_circ_scl"+"/eicu-crd/preprocessing_data"
root_dir = "/Users/DAHS/Desktop/early_prediction_of_circ_scl"


def generate_adm():
    data=pd.read_csv(local+"/cohort/cohort_.csv.gz", compression='gzip', header=0, index_col=None)
    data['los(min)']=data['unitdischargeoffset']-data['unitadmissionoffset']
    data['los(hour)']=np.rint(data['los(min)']/60)
    
    data=data[data['los(hour)']>0]
    data['Age']=data['Age'].astype(int)

    return data

def generate_proc(data):
    proc=pd.read_csv(local+ "/features/preproc_proc(selected)_icu.csv.gz", compression='gzip', header=0, index_col=None)
    proc=proc[proc['patientunitstayid'].isin(data['patientunitstayid'])]
    
    proc['start_time'] = np.rint(proc['event_time_from_admit']/60)
    proc=proc.drop(columns=['event_time_from_admit'])
    proc=proc[proc['start_time']>=0]
    
    ###Remove where event time is after discharge time
    proc=pd.merge(proc, data[['patientunitstayid','los(hour)']],on='patientunitstayid',how='left')
    proc['sanity']=proc['los(hour)']-proc['start_time']
    proc=proc[(proc['sanity']>0)&~(proc['treatmentstring']=='Ventilator')] #using only ventliator event
    del proc['sanity']
    
    proc = proc[['uniquepid','patientunitstayid','treatmentid', 'treatmentstring', 'start_time']]
    
    return proc

def generate_out(data):
    out=pd.read_csv(local+ "/features/preproc_out(selected)_icu.csv.gz", compression='gzip', header=0, index_col=None)
    out=out[out['patientunitstayid'].isin(data['patientunitstayid'])]
    
    out['start_time']=np.rint(out['event_time_from_admit']/60)
    out=out.drop(columns=['event_time_from_admit'])
    out=out[out['start_time']>=0]
    
    ###Remove where event time is after discharge time
    out=pd.merge(out,data[['patientunitstayid','los(hour)']],on='patientunitstayid',how='left')
    out['sanity']=out['los(hour)']-out['start_time']
    out=out[out['sanity']>0]
    del out['sanity']
    
    return out

def generate_chart(data):
    chunksize = 5000000
    final=pd.DataFrame()
    for chart in tqdm(pd.read_csv(local+ "/features/preproc_chart(selected)_icu.csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
        chart=chart[chart['patientunitstayid'].isin(data['patientunitstayid'])]
   
        chart['start_time']=np.rint(chart['event_time_from_admit']/60)
        chart=chart.drop(columns=['event_time_from_admit'])
        chart=chart[chart['start_time']>=0]

        ###Remove where event time is after discharge time
        chart=pd.merge(chart,data[['patientunitstayid','los(hour)']],on='patientunitstayid',how='left')
        chart['sanity']=chart['los(hour)']-chart['start_time']
        chart=chart[chart['sanity']>0]
        del chart['sanity']
        del chart['los(hour)']
        
        if final.empty:
            final=chart
        else:
            final=final.append(chart, ignore_index=True)
    
    return final

def generate_labs(data):
    chunksize = 10000000
    final=pd.DataFrame()
    for labs in tqdm(pd.read_csv(local+ "/features/preproc_labs(selected).csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
        labs=labs[labs['patientunitstayid'].isin(data['patientunitstayid'])]
        labs['start_time']=np.rint(labs['lab_time_from_admit']/60)
        labs=labs.drop(columns=['lab_time_from_admit'])
        labs=labs[labs['start_time']>=0]

        ###Remove where event time is after discharge time
        labs=pd.merge(labs,data[['patientunitstayid','los(hour)']],on='patientunitstayid',how='left')
        labs['sanity']=labs['los(hour)']-labs['start_time']
        labs=labs[labs['sanity']>0]
        del labs['sanity']
        
        if final.empty:
            final=labs
        else:
            final=final.append(labs, ignore_index=True)

    return final

def generate_total_procedure(procedure, data): # micro event + procedure
    microlabs=pd.read_csv(local+ "/features/premicrolabs_microlabs(selected).csv.gz", compression='gzip', header=0, index_col=None)
    microlabs=microlabs[microlabs['patientunitstayid'].isin(data['patientunitstayid'])]
    
    microlabs['start_time'] = np.rint(microlabs['lab_time_from_admit']/60)
    microlabs=microlabs.drop(columns=['lab_time_from_admit'])
    microlabs=microlabs[microlabs['start_time']>=0]
    
    ###Remove where event time is after discharge time
    microlabs=pd.merge(microlabs, data[['patientunitstayid','los(hour)']],on='patientunitstayid',how='left')
    microlabs['sanity']=microlabs['los(hour)']-microlabs['start_time']
    microlabs=microlabs[microlabs['sanity']>0]
    del microlabs['sanity']
    
    microlabs = microlabs[['uniquepid','patientunitstayid','labid', 'labname', 'start_time']]
    microlabs = microlabs.rename(columns = {'labid':'treatmentid', 'labname': 'treatmentstring'})
    
    total_proc = pd.concat([microlabs, procedure], ignore_index = True)
    
    return total_proc


def generate_meds(data):
    meds=pd.read_csv(local+ "/features/preproc_med(selected)_icu.csv.gz", compression='gzip', header=0, index_col=None)

    meds['start_time']=np.rint(meds['start_hours_from_admit']/60)

    meds['stop_time'] = meds['start_time'] + 1
    meds=meds.drop(columns=['start_hours_from_admit', 'stop_hours_from_admit'])
    #####Sanity check
    meds['sanity']=meds['stop_time']-meds['start_time']
    meds=meds[meds['sanity']>0]
    del meds['sanity']
    #####Select hadm_id as in main file
    meds=meds[meds['patientunitstayid'].isin(data['patientunitstayid'])]
    meds=pd.merge(meds,data[['patientunitstayid','los(hour)']],on='patientunitstayid',how='left')

    #####Remove where start time is after end of visit
    meds['sanity']=meds['los(hour)']-meds['start_time']
    meds=meds[meds['sanity']>0]
    del meds['sanity']
    ####Any stop_time after end of visit is set at end of visit
    meds.loc[meds['stop_time'] > meds['los(hour)'],'stop_time']=meds.loc[meds['stop_time'] > meds['los(hour)'],'los(hour)']
    del meds['los(hour)']
    
    meds['cellvaluenumeric']=meds['cellvaluenumeric'].apply(pd.to_numeric, errors='coerce')
    
    return meds


def generate_ing(data):
    ing=pd.read_csv(local+ "/features/preproc_ing(selected)_icu.csv.gz", compression='gzip', header=0, index_col=None)

    ing['start_time']=np.rint(ing['start_hours_from_admit']/24)
    ing['stop_time']=ing['start_time'] + 1
    ing['amount']=ing['drugrate'].apply(pd.to_numeric, errors='coerce')
    
    ing=ing.drop(columns=['start_hours_from_admit', 'stop_hours_from_admit', 'drugrate'])
    #####Sanity check
    ing['sanity']=ing['stop_time']-ing['start_time']
    ing=ing[ing['sanity']>0]
    del ing['sanity']
    #####Select hadm_id as in main file
    ing=ing[ing['patientunitstayid'].isin(data['patientunitstayid'])]
    ing=pd.merge(ing,data[['patientunitstayid','los(hour)']],on='patientunitstayid',how='left')

    #####Remove where start time is after end of visit
    ing['sanity']=ing['los(hour)']-ing['start_time']
    ing=ing[ing['sanity']>0]
    del ing['sanity']
    ####Any stop_time after end of visit is set at end of visit
    ing.loc[ing['stop_time'] > ing['los(hour)'],'stop_time']=ing.loc[ing['stop_time'] > ing['los(hour)'],'los(hour)']
    del ing['los(hour)']
    
    return ing


def generate_vent(data):
    vent=pd.read_csv(local+ "/features/preproc_vent_icu.csv.gz", compression='gzip', header=0, index_col=None)

    vent['start_time']=np.rint(vent['ventstartoffset']/24)
    vent['stop_time']=np.rint(vent['ventendoffset']/24)

    vent=vent.drop(columns=['ventstartoffset', 'ventendoffset'])
    #####Sanity check
    vent['sanity']=vent['stop_time']-vent['start_time']
    vent=vent[vent['sanity']>0]
    del vent['sanity']
    #####Select hadm_id as in main file
    vent=vent[vent['patientunitstayid'].isin(data['patientunitstayid'])]
    vent=pd.merge(vent,data[['patientunitstayid','los(hour)']],on='patientunitstayid',how='left')

    #####Remove where start time is after end of visit
    vent['sanity']=vent['los(hour)']-vent['start_time']
    vent=vent[vent['sanity']>0]
    del vent['sanity']
    ####Any stop_time after end of visit is set at end of visit
    vent.loc[vent['stop_time'] > vent['los(hour)'],'stop_time']=vent.loc[vent['stop_time'] > vent['los(hour)'],'los(hour)']
    del vent['los(hour)']
    
    return vent