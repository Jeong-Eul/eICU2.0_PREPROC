import os 
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from pathlib import Path
import os
import importlib
import warnings
import pdb

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
    microlabs=pd.read_csv(local+ "/features/preproc_microlabs(selected).csv.gz", compression='gzip', header=0, index_col=None)
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

    ing['start_time']=np.rint(ing['start_hours_from_admit']/60)
    ing['stop_time']=np.ceil(ing['stop_hours_from_admit']/60)
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

    vent['start_time']=np.rint(vent['ventstartoffset']/60)
    vent['stop_time']=np.ceil(vent['ventendoffset']/60)

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


def tabularization(feat_med, feat_ing, feat_out, feat_chart, feat_lab, feat_vent, feat_proc,
                   final_meds, final_ing, final_proc, final_out, final_chart, final_labs, final_vent, valid_stay_ids, data):
    
    print("# Unique gender: ", data.gender.nunique())
    print("# Unique ethnicity: ", data.ethnicity.nunique())
    print("=====================")
    print('Number of patient: ', len(data.uniquepid.unique()))
    print('Number of stay: ', len(data.patientunitstayid.unique()))
    print('Expected value of observation: ', data['los(hour)'].sum())
    print("=====================")
    print()
    
    
    for hid in tqdm(valid_stay_ids, desc = 'Tabularize EHR for total stay 16,120'):
        grp=data[data['patientunitstayid']==hid]
        los = int(grp['los(hour)'].values)
        height = grp['admissionheight'].values[0]
        weight = grp['admissionweight'].values[0]
        if not os.path.exists(local+"/csv/"+str(hid)):
            os.makedirs(local+"/csv/"+str(hid))
        
        dyn_csv=pd.DataFrame()
        
        ###MEDS
        if(feat_med):
            feat=final_meds['celllabel'].unique()
            df2=final_meds[final_meds['patientunitstayid']==hid]
            if df2.shape[0]==0:
                dose=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                dose=dose.fillna(0)
                dose.columns=pd.MultiIndex.from_product([["MEDS"], dose.columns])
            else:
                dose=df2.pivot_table(index='start_time',columns='celllabel',values='cellvaluenumeric')
                df2=df2.pivot_table(index='start_time',columns='celllabel',values='stop_time') #value는 큰 의미 없음

                add_indices = pd.Index(range(los)).difference(df2.index) # 처방 된 시간과 los 비교 후 처방이 이루어지지 않은 시간 포인트만큼 시간 인덱스 생성
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan) 
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.ffill()
                df2=df2.fillna(0)

                dose=pd.concat([dose, add_df])
                dose=dose.sort_index()
                dose=dose.ffill()
                dose=dose.fillna(0)
    
                df2.iloc[:,0:]=df2.iloc[:,0:].sub(df2.index,0) #end time - start time
                df2[df2>0]=1 # 약물 처방 시간 만큼 1
                df2[df2<0]=0 #약을 처방 받지 않은 경우에는 0으로 채웠었기 때문에 sub 연산시 음 값을 가지게 되어 0으로 변환됨
        
                dose.iloc[:,0:]=df2.iloc[:,0:]*dose.iloc[:,0:] # 실제 처방 된 경우의 값만 살아 남음
                feat_df=pd.DataFrame(columns=list(set(feat)-set(dose.columns)))
                dose=pd.concat([dose,feat_df],axis=1)


                dose=dose[feat]
                dose=dose.fillna(0)
                dose.columns=pd.MultiIndex.from_product([["MEDS"], dose.columns])
        
                
            if(dyn_csv.empty):
                dyn_csv=dose
            else:
                dyn_csv=pd.concat([dyn_csv,dose],axis=1)
            
        
        ###INGS
        if(feat_ing):
            feat=final_ing['drugname'].unique()
            feat_rate = [item + '_rate' for item in feat]
            df2=final_ing[final_ing['patientunitstayid']==hid]
            if df2.shape[0]==0:
                amount=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                amount=amount.fillna(0)
                amount.columns=pd.MultiIndex.from_product([["INGS"], amount.columns])
                
                rate=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat_rate)
                rate=rate.fillna(0)
                rate.columns=pd.MultiIndex.from_product([["RATE"], rate.columns])
            else:
                amount=df2.pivot_table(index='start_time',columns='drugname',values='amount')
                rate=df2.pivot_table(index='start_time',columns='drugname',values='infusionrate')
                df2=df2.pivot_table(index='start_time',columns='drugname',values='stop_time')
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.ffill()
                df2=df2.fillna(0)

                amount=pd.concat([amount, add_df])
                amount=amount.sort_index()
                amount=amount.ffill()
                amount=amount.fillna(0)
                

                
                rate=pd.concat([rate, add_df])
                rate=rate.sort_index()
                rate=rate.ffill()
                rate=rate.fillna(-1)
            
                df2.iloc[:,0:]=df2.iloc[:,0:].sub(df2.index,0)
                df2[df2>0]=1
                df2[df2<0]=0

                rate.iloc[:,0:]=df2.iloc[:,0:]*rate.iloc[:,0:]
                amount.iloc[:,0:]=df2.iloc[:,0:]*amount.iloc[:,0:]
                
                feat_df=pd.DataFrame(columns=list(set(feat)-set(amount.columns)))
                feat_df_rate=pd.DataFrame(columns=list(set(feat_rate)-set(rate.columns)))
                
                amount=pd.concat([amount,feat_df],axis=1)
                rate = pd.concat([rate, feat_df_rate], axis=1)

                amount=amount[feat]
                rate=rate[feat_rate]
                
                amount=amount.fillna(0)
                rate=rate.fillna(0)
                
                amount.columns=pd.MultiIndex.from_product([["INGS"], amount.columns])
                rate.columns=pd.MultiIndex.from_product([["RATE"], rate.columns])
                
            if(dyn_csv.empty):
                dyn_csv= pd.concat([amount, rate],axis=1)
            else:
                ingredient = pd.concat([amount, rate],axis=1)
                dyn_csv=pd.concat([dyn_csv,ingredient],axis=1)
            
            
        ###PROCS
        if(feat_proc):
            feat = final_proc['treatmentstring'].unique()
            df2 = final_proc[final_proc['patientunitstayid']==hid]
            
            if df2.shape[0]==0:
                df2=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                df2=df2.fillna(0)
                df2.columns=pd.MultiIndex.from_product([["PROC"], df2.columns])
            else:
                df2['val']=1
                #print(df2)
                df2=df2.pivot_table(index='start_time',columns='treatmentstring',values='val')
                #print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)
                df2[df2>0]=1

                feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                df2=pd.concat([df2,feat_df],axis=1)

                df2=df2[feat]
                df2=df2.fillna(0)
                df2.columns=pd.MultiIndex.from_product([["PROC"], df2.columns])
            
            if(dyn_csv.empty):
                dyn_csv=df2
            else:
                dyn_csv=pd.concat([dyn_csv,df2],axis=1)
            
            
        ###OUT
        if(feat_out):
            feat=final_out['celllabel'].unique()
            df2=final_out[final_out['patientunitstayid']==hid]
        
            if df2.shape[0]==0:
                val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                val=val.fillna(0)
                val.columns=pd.MultiIndex.from_product([["OUT"], val.columns])
            else:
                val=df2.pivot_table(index='start_time',columns='celllabel', values='cellvaluenumeric')
                df2['val']=1
                df2=df2.pivot_table(index='start_time',columns='celllabel',values='val')

                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)

                val=pd.concat([val, add_df])
                val=val.sort_index()
                val=val.fillna(0)
                
                df2[df2>0]=1
                df2[df2<0]=0

                feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                val=pd.concat([val,feat_df],axis=1)

                val=val[feat]
                val=val.fillna(0)
                val.columns=pd.MultiIndex.from_product([["OUT"], val.columns])
            
            if(dyn_csv.empty):
                dyn_csv=val
            else:
                dyn_csv=pd.concat([dyn_csv,val],axis=1)
                
            
        ###CHART
        if(feat_chart):
            feat=final_chart['nursingchartcelltypevallabel'].unique()
            df2=final_chart[final_chart['patientunitstayid']==hid]
            if df2.shape[0]==0:
                val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                val=val.fillna(0)
                val.columns=pd.MultiIndex.from_product([["CHART"], val.columns])
            else:
                val=df2.pivot_table(index='start_time',columns='nursingchartcelltypevallabel',values='nursingchartvalue')
                df2['val']=1
                df2=df2.pivot_table(index='start_time',columns='nursingchartcelltypevallabel',values='val')
                #print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)

                val=pd.concat([val, add_df])
                val=val.sort_index()
                
                ## ECMO
                if 'ECMO' in val.columns:
                    val['ECMO'] = val['ECMO'].notna().astype(int)
                
                ## MAP
                if 'MAP' in val.columns:
                    map_column = val['MAP'].copy()
                    val = val.drop(columns=['MAP']).ffill()
                    map_column = map_column.fillna((val['ABPs'] + 2*val['ABPd'])/3)
                    val['MAP'] = map_column
                    
                else:
                    val = val.ffill()
                    val['MAP'] = (val['ABPs'] + 2*val['ABPd'])/3
                
        
                df2[df2>0]=1
                df2[df2<0]=0

                feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                val=pd.concat([val,feat_df],axis=1)

                val=val[feat]
                val.columns=pd.MultiIndex.from_product([["CHART"], val.columns])
            
            if(dyn_csv.empty):
                dyn_csv=val
            else:
                dyn_csv=pd.concat([dyn_csv,val],axis=1)
        
        ###LABS
        if(feat_lab):
            feat=final_labs['labname'].unique()
            df2=final_labs[final_labs['patientunitstayid']==hid]
            if df2.shape[0]==0:
                val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                val=val.fillna(0)
                val.columns=pd.MultiIndex.from_product([["LAB"], val.columns])
            else:
                val=df2.pivot_table(index='start_time',columns='labname',values='labresult')
                df2['val']=1
                df2=df2.pivot_table(index='start_time',columns='labname',values='val')
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)

                val=pd.concat([val, add_df])
                val=val.sort_index()
                
                val=val.ffill()

                df2[df2>0]=1
                df2[df2<0]=0
                
                feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                val=pd.concat([val,feat_df],axis=1)

                val=val[feat]
                val.columns=pd.MultiIndex.from_product([["LAB"], val.columns])
            
            if(dyn_csv.empty):
                dyn_csv=val
            else:
                dyn_csv=pd.concat([dyn_csv,val],axis=1)
                
                
        
        if(feat_vent):
            feat=final_vent['label'].unique()
            df2=final_vent[final_vent['patientunitstayid']==hid]
            if df2.shape[0]==0:
                amount=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                amount=amount.fillna(0)
                amount.columns=pd.MultiIndex.from_product([["VENT"], amount.columns])
            else:
                start_time = int(df2['start_time'].iloc[0])
                stop_time = int(df2['stop_time'].iloc[0])
                time_range = range(start_time, stop_time)

                amount = pd.DataFrame(index=time_range)
                amount['Ventilator'] = 1

                add_indices = pd.Index(range(los)).difference(amount.index)
                add_df = pd.DataFrame(index=add_indices, columns=amount.columns).fillna(np.nan)

                amount=pd.concat([amount, add_df])
                amount=amount.sort_index()
                amount=amount.fillna(0)
                
                feat_df=pd.DataFrame(columns=list(set(feat)-set(amount.columns)))
                amount=pd.concat([amount,feat_df],axis=1)

                amount=amount[feat]
                amount=amount.fillna(0)
                
                amount.columns=pd.MultiIndex.from_product([["VENT"], amount.columns])
                
            if(dyn_csv.empty):
                dyn_csv=amount
            else:
                dyn_csv=pd.concat([dyn_csv,amount],axis=1)
        
        #[ ====== Save temporal data to csv ====== ]
        
        dyn_csv['height'] = height
        dyn_csv['weight'] = weight
      
        dyn_csv.to_csv(local+'/csv/'+str(hid)+'/dynamic_proc.csv',index=False)
        
    print("[ SUCCESSFULLY SAVED TOTAL UNIT STAY DATA ]")