import numpy as np
import pandas as pd

from collections import defaultdict

def col_type(series):
    
    '''identify type of data of a column/series'''
    
    tclass=type(series[0])
    dtype=tclass.__name__

    return dtype

def nulls_num(series):
    
    '''find missing values in numerical or boolean series'''
    
    return series.isnull()

def nulls_cat(series):
    
    '''find various types of missing conditions in categories'''
    
    cond0=series.isnull()
    cond1=series.str.startswith('-')
    cond2=series.str.startswith('0')
    cond3=series.str.startswith('?')
    cond4=series.str.startswith('none')
    return [cond0,cond1,cond2,cond3,cond4]    

def nulls_perc(series):
    
    ''' percentage and total number of missing values in a column'''

    values=series.values
    if col_type(series)=='str': 
        total=np.sum([nulls_cat(series)[i].sum() for i in range(len(nulls_cat(series)))])
    else: total=nulls_num(series).sum()  
    perc=round(total/len(values)*100,1)
    return total,perc    


def dataFr_info(dataFr):
    
    '''create dictionary of data types in a dataframe
    print information on percentage of missing values'''
    
    types_dict=defaultdict(list)
    flag=0
    
    for col in dataFr.columns:
        key=col_type(dataFr[col])
        types_dict[key].append(col)
        nulls=nulls_perc(dataFr[col])
        if nulls[1]==0: continue
        print('Percentage of missing values in {}:'.format(col),nulls[1])
        flag=1
        
    if flag==0: print('There are no missing values in the dataframe')    
    print('\n')    
    
    for k,v in types_dict.items(): print('Number of columns of {} type:'.format(k),len(v),'\n',v,'\n')   
        
    return types_dict 


def drop_columns(dataFr,perc=40):
    
    '''drop columns of a dataframe if they have more then a certain 
    percentage of missing values
    or if the value is unique'''
    
    print('Number of columns before drop:',len(dataFr.columns.values))
    dropped_cols=[]
    newDf=pd.DataFrame()
    for col in dataFr.columns.values:
        cond1=nulls_perc(dataFr[col])[1]>perc
        #cond2=len(dataFr[col].unique())==1
        tot=len(dataFr[col])
        maj=dataFr[col].value_counts().apply(lambda val: val/tot)>0.75
        cond2=maj.sum()==1
        
        if cond1 or cond2:
            dropped_cols.append(col)

    newDf=dataFr.drop(dropped_cols,axis=1)
    #print('Number of dropped columns:',len(dropped_cols))
    if len(dropped_cols)!=0: 
        print('Number of columns after drop:',len(dataFr.columns.values)-len(dropped_cols))
        print('Dropped columns:',dropped_cols)
    else: print('No columns were dropped')    
    return(newDf)    
