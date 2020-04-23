from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.ensemble import RandomSurvivalForest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepare(X,y):
    
    '''reshape data to be used by the various models from Scikit-survival package'''
    
    # Kaplan Meier requires only an event column and a duration
    E=y.copy().astype('bool')
    T= X['duration']
    
    # Cox and SVM require the target to have 2 columns, the first one being the event, the second one the duration
    Xc=X.copy()
    #Xc.drop(columns=['construction_year'],inplace=True,axis=1)
    #y1=y.copy().reset_index(drop=True)
    #l=[(e,t) for e,t in zip(y1,Xc['duration'])]
    y1=y.copy()
    l=[(e,t) for e,t in zip(y1,Xc['duration'])]
    yc=np.array(l,'bool,i')
    Xc.drop(columns=['duration'],axis=1,inplace=True)
    
    # Random Survival Forest requires the second column in the target to be in double format
    # and the features to be in a structure format
    Xrf=np.column_stack(Xc.copy().astype('float').values).T
    yrf=np.array(l,'bool,double')
    
    return E,T,Xc,yc,Xrf,yrf


def surv_data(X,y):
    
    '''reshape data to be used by the various models from Scikit-survival package'''

    #y=y.apply(lambda val: 1 if val!=0 else 0)
    #X.reset_index(drop=True,inplace=True)
    #y.reset_index(drop=True,inplace=True)
    # once the duration is defined, the year features can be dropped, since it would be a redundant information
    X['duration']=X['recorded_year']-X['construction_year']
    X.drop(columns=['recorded_year','construction_year'],axis=1,inplace=True)
    itd=X[X['duration']<0].index
    X.drop(itd,axis=0,inplace=True)
    y.drop(itd,axis=0,inplace=True)
    
    # Kaplan Meier requires only an event column and a duration
    E=y.copy().astype('bool')
    T= X['duration']
    
    # Cox and SVM require the target to have 2 columns, the first one being the event, the second one the duration
    Xc=X.copy()
    y1=y.copy()
    l=[(e,t) for e,t in zip(y1,Xc['duration'])]
    yc=np.array(l,'bool,i')
    Xc.drop(columns=['duration'],axis=1,inplace=True)
    
    # Random Survival Forest requires the second column in the target to be in double format
    # and the features to be in a structure format
    Xrf=np.column_stack(Xc.copy().astype('float').values).T
    yrf=np.array(l,'bool,double')
    
    return E,T,Xc,yc,Xrf,yrf

    
def km(event,duration):
        
    '''Kaplan Meir estimator'''
    
    years, survival_prob = kaplan_meier_estimator(event,duration)
    plt.figure(figsize=(10,6))
    plt.step(years, survival_prob,where='post')
    plt.title('Estimated probability of survival',fontsize=16)
    plt.ylabel("$\hat{S}(t)$")
    plt.xlabel("years")
    
    
def cox(X,y):
    
    '''Cox regression model'''
    
    cph = CoxPHSurvivalAnalysis()
    cox = cph.fit(X, y)    
    
    print('Cox score:', cox.score(X, y))
    print('Features coefficients:')
    coeff=pd.Series(cox.coef_, index=X.columns)
    print(coeff)
    
    
def ssvm(X,y):
    
    '''Support Vectors applied to survival analyses'''
    
    svm = FastSurvivalSVM(optimizer="rbtree", max_iter=1000, tol=1e-6, random_state=0)
    svmf=svm.fit(X, y)
    print('Svm score:', svmf.score(X, y))
    
        
def srf(X,y,random):
    
    '''Random Forest applied to survival analyses'''    

    rsf = RandomSurvivalForest(n_estimators=50,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=random)
    rsf.fit(X, y)
    print('Random Forest score:', rsf.score(X, y))

    
    