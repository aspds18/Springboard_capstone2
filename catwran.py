from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import itertools
import category_encoders as ce


import pandas as pd
import scipy.stats as scs
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import infodf as idf

def fill_missing(series):
    
    '''fill missing values of a categorical variable'''
    
    #first replace nan
    series.fillna('unknown',inplace=True)
    cond_list=idf.nulls_cat(series)
    for cond in cond_list[1:]: series[cond]='unknown'
        
def reduce_cat(series,num=21):
    
    '''replace entry that is not in the top num with the word "other" '''
    
    if len(series.unique())>=num: top=num
    else: top=len(series.unique())    
       
    common=series.value_counts()[:top].index.tolist()
    print(common)
    series=series.apply(lambda val: 'Other' if val not in common else val)
    
    return series        

def encoder(dataFr):
    
    '''label encoding of dataframe of categorical features
    use defaultdict as suggested here:
    https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn'''
    
    d = defaultdict(LabelEncoder)
    dataFr_enc = dataFr.apply(lambda x: d[x.name].fit_transform(x))
    dataFr_noenc=dataFr_enc.apply(lambda x: d[x.name].inverse_transform(x))
    
    return dataFr_enc,dataFr_noenc

def nom_encoder(dFr,y,enc):
    
    '''encoding a dataframe of categorical features
    use encoders from category_encoders package'''
    
    dFr_enc=enc.fit_transform(dFr,y)
    return dFr_enc


def encode_bool(series):
    
    '''encode series of boolean to float,
    fill missing values'''
    
    series.replace({True:1, False:0},inplace=True)
    series.fillna(2,inplace=True)

    return series


def conf_matrix(ser1,ser2):
    
    '''create confusion matrix used for correlations'''
    
    mat=pd.crosstab(ser1,ser2).values
    return mat
    
def cramers_corrected_stat(ser1,ser2):
    
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
        also suggested here:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix 
    """
    
    mat=conf_matrix(ser1,ser2)
    chi2 = float(scs.chi2_contingency(mat)[0])
    n = float(mat.sum())
    phi2 = chi2/n
    r,k = mat.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

# this is the original Cramers V statistic formula
def cramers_stat(ser1,ser2):
    
    mat=conf_matrix(ser1,ser2)
    chi2 = scs.chi2_contingency(mat)[0]
    n = mat.sum()
    return np.sqrt(chi2 / (n*(min(mat.shape)-1)))

def cat_correlation(df,stat=cramers_corrected_stat):
    
    '''plot correlations heatmap'''
    
    cols=df.columns.values.tolist()
    corrM = np.ones((len(df.columns.values),len(df.columns.values)))

    for col1, col2 in itertools.combinations(cols, 2):
        idx1, idx2 = cols.index(col1), cols.index(col2)
        corrM[idx1, idx2] = round(cramers_stat(df[col1], df[col2]),2)
        corrM[idx2, idx1] = corrM[idx1, idx2]
        corrM[idx1,idx1]=1

    corr = pd.DataFrame(corrM, index=df.columns, columns=df.columns)

    
    _ = plt.figure(figsize=(14,10))
    _ = plt.title('Cramers V correlation between variables',fontsize=16)
    _ = sns.heatmap(corr,linewidths=0.5,cmap="YlGnBu", annot=True)
    _ = plt.yticks(rotation=0)
    
    