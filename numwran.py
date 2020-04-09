import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def q25(series):
    return series.quantile(25/100)

def q75(series):
    return series.quantile(75/100)


def num_stats(col):
    
    '''main stats of numerical variable'''
    
    values=col.values
    avg=round(np.mean(values))
    md=np.median(values)
    std=round(np.std(values))
    per25=q25(col)
    per75=q75(col)
    return avg,md,std,per25,per75

    
def print_stats(col):
    
    avg,md,std,per25,per75=num_stats(col)
    print('Mean:',avg,'Median:',md,'','Std:',std,'','Q25:',per25,'','Q75:',per75)
    
    
def print_zero(df,col_list):
    
    '''print columns' total of zero values and percentage''' 
    
    obs=len(df)
    
    for col in col_list:
        
        total = (df[col]==0).sum()
        
        perc_zero=total, round(total/obs*100)
        print('Total, percentage of zeros in {}: '.format(col),perc_zero)    
        
        
def correlations(features,method='pearson'):
    
    '''plot heatmap of correlation between features'''
    
    corr=features.corr(method=method)
    _ = plt.figure(figsize=(18,12))
    _ = sns.heatmap(corr,annot=True,fmt='.2f',square=True,cmap="YlGnBu",linewidths=0.5)        