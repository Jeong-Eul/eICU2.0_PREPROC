import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

local = "/Users/DAHS/Desktop/early_prediction_of_circ_scl"+"/eicu-crd/preprocessing_data"

class Integration_data():
    def __init__(self):
        self.categorical_encoding()
        
    def create_stay_id(self):
        data=pd.read_csv(local+'/demo.csv', index_col = 0)

        hids=data['patientunitstayid'].unique()
        print("Total stay",len(hids))
        return data, hids
    
    def categorical_encoding(self):
        data, hids=self.create_stay_id()
        data=self.getdata(data, hids)
        return data
        
            
    def getdata(self,dataset, ids):
        df_list = []   
        for sample in tqdm(ids):
            dyn=pd.read_csv(local+'/csv/'+str(sample)+'/dynamic_proc.csv',header=[0,1])
            stat = dataset[dataset['patientunitstayid']==int(sample)]
            
            dyn.columns=dyn.columns.droplevel(0)
            columns_to_copy = ['uniquepid', 'patientunitstayid', 'Age', 'gender', 'ethnicity']
            for column in columns_to_copy:
                dyn[column] = stat[column].values[0]
                
            df_list.append(dyn)
            
        df = pd.concat(df_list, axis = 0)
        
        print("total stay dataframe shape",df.shape)
        
        #encoding categorical
        gen_encoder = LabelEncoder()
        eth_encoder = LabelEncoder()

        gen_encoder.fit(df['gender'])
        eth_encoder.fit(df['ethnicity'])

        df['gender']=gen_encoder.transform(df['gender'])
        df['ethnicity']=eth_encoder.transform(df['ethnicity'])
        
        df.to_csv('Total.csv')
        return df