from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.ensemble import RandomSurvivalForest

from sklearn.model_selection import train_test_split,GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def xy(dataFr):
    
    '''create features and target for this specific project'''
    
    y=dataFr['status_group']
    y=y.apply(lambda val: 1 if val!=0 else 0)
    X=dataFr.drop(columns=['status_group'])
    return X,y


def split_data(X,y,random=9,size=0.3):
    
    '''split data into training and test set'''
    
    # stratify keeps the proportions of labels 
    Xtr, Xtst, ytr, ytst = train_test_split(X,y,test_size=size,random_state=random, stratify=y)
    return Xtr,Xtst,ytr,ytst


def surv_data(X,y):
    
    '''reshape data to be used by the various models from Scikit-survival package'''

    # the time to event or duration is needed in the survival models
    # once the duration is defined, the year features can be dropped, since it would be a redundant information
    X=X.assign(duration=-X.loc[:,'construction_year']+X.loc[:,'recorded_year'])
    X.drop(columns=['recorded_year','construction_year'],axis=1,inplace=True)
    itd=X[X['duration']<0].index
    X.drop(itd,axis=0,inplace=True)
    y.drop(itd,axis=0,inplace=True)
    
    # Kaplan Meier requires only an event column and a duration
    E=y.astype('bool')
    T= X['duration']
    
    # Cox and SVM require the target to have 2 columns, the first one being the event, the second one the duration
    Xc=X
    l=[(e,t) for e,t in zip(y,X['duration'])]
    yc=np.array(l,'bool,i')
    Xc.drop(columns=['duration'],axis=1,inplace=True)
    
    # Random Survival Forest requires the second column in the target to be in double format
    # and the features to be in a structure format
    Xrf=np.column_stack(Xc.astype('float').values).T
    yrf=np.array(l,'bool,double')
    
    return E,T,Xc,yc,Xrf,yrf


def tune_model(base,Xtr,ytr,Xtst,ytst,grid):
    
    '''grid search of a base model'''
    
    surv_cv= GridSearchCV(base, grid, cv=5,refit=True)
    surv_cv.fit(Xtr,ytr)
    print("Tuned Parameters: {}".format(surv_cv.best_params_)) 
    print('')
    
    best=surv_cv.best_estimator_
    print('Training score:',round(best.score(Xtr,ytr),3))
    print('Test score:',round(best.score(Xtst,ytst),3))
   
    return best



# Kaplan Meier section#    
def km(event,duration):
        
    '''Kaplan Meier estimator'''
    
    years, survival_prob = kaplan_meier_estimator(event,duration)
    plt.figure(figsize=(10,6))
    plt.step(years, survival_prob,where='post')
    plt.title('Estimated probability of survival',fontsize=16)
    plt.ylabel("$\hat{S}(t)$",fontsize=14)
    plt.xlabel("years",fontsize=14)
    
def prob_vs_feature(df,feature,E,T):
    
    '''plot survival probabilty with respect to a specific feature'''
    
    plt.figure(figsize=(10,6))
    plt.title('Estimated probability of survival by {}'.format(feature),fontsize=16)
    for ft in df[feature].unique().tolist():
        mask = df[feature] == ft
        time, survival_prob = kaplan_meier_estimator(
            E[mask],
            T[mask])
    
        plt.step(time, survival_prob, where="post",
             label=ft)
    
    plt.ylabel("$\hat{S}(t)$",fontsize=14)
    plt.xlabel("years",fontsize=14)
    plt.legend(loc="best")

    
# Cox Regression section#    
def cox(X,y):
    
    '''Cox regression model'''
    
    cph = CoxPHSurvivalAnalysis()
    cox = cph.fit(X, y)    
    
    print('Cox score:', cox.score(X, y))
    print('Features coefficients:')
    coeff=pd.Series(cox.coef_, index=X.columns)
    print(coeff)
    return cox

def cox_prediction(index,model,X):
    
    '''make prediction using a trained Cox regression model'''
    
    Xp=X.iloc[index]
    surv=model.predict_survival_function(Xp)
    haz=model.predict_cumulative_hazard_function(Xp)
    return surv,haz
    
def plot_cox_prediction(model,sur,haz):
    
    '''plot the survival probability and hazard function of Cox model'''
    
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    plt.subplots_adjust(wspace=0.35)
    fig.subplots_adjust(top=0.9)
    
    plt.subplot(1, 2, 1)
    plt.title("Survival probability ",fontsize=16)
    plt.xlabel("Years",fontsize=14)
    for i, c in enumerate(sur):
        plt.step(c.x, c.y, where="post")
                
    plt.subplot(1,2,2)    
    plt.title("Cumulative hazard",fontsize=16)
    plt.xlabel("Years",fontsize=14)
    for i, c in enumerate(haz):
        plt.step(c.x, c.y, where="post")
    
# Support vector machine section    
def ssvm(X,y):
    
    '''Support Vectors applied to survival analyses'''
    
    svm = FastSurvivalSVM(optimizer="rbtree", max_iter=1000, tol=1e-6, random_state=0)
    svmf=svm.fit(X, y)
    print('Svm score:', svmf.score(X, y))
    
    return svmf
    
# Random Survival Forest section        
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
    
    return rsf

def rf_prediction(index,model,X):
    
    '''make prediction using a trained Random Survival Forest'''
    
    #X is the array of features, index is the observation
    Xp=X[index].reshape(-1,1).T
    surv=model.predict_survival_function(Xp)
    haz=model.predict_cumulative_hazard_function(Xp)
    
    return surv,haz

    
def plot_rf_prediction(model,sur,haz):
    
    '''plot the survival probability and hazard function of Random Forest'''
    
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    plt.subplots_adjust(wspace=0.35)
    fig.subplots_adjust(top=0.9)
    plt.subplot(1, 2, 1)
    plt.step(model.event_times_, sur.T, where="post")
    plt.title("Survival probability",fontsize=16)
    plt.xlabel("Years",fontsize=14)
    plt.subplot(1, 2, 2)
    plt.step(model.event_times_, haz.T, where="post")
    plt.title("Cumulative hazard",fontsize=16)
    plt.xlabel("Years",fontsize=14)   
    
    
    
def plot_predictions(rf,rf_sur,rf_haz,cox_sur,cox_haz):
    
    '''compare survival probability and hazard function of Random Survival Forest and Cox'''
    
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    plt.subplots_adjust(wspace=0.35)
    fig.subplots_adjust(top=0.9)
    plt.subplot(1, 2, 1)
    plt.title("Survival probability ",fontsize=16)
    plt.xlabel("Years",fontsize=14)
    
    plt.step(rf.event_times_, rf_sur.T, where="post",label='Random Forest')
    for i, c in enumerate(cox_sur):
        plt.step(c.x, c.y, where="post",label='Cox')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Cumulative hazard",fontsize=16)
    plt.xlabel("Years",fontsize=14)
    
    plt.step(rf.event_times_, rf_haz.T, where="post",label='Random Forest')
    for i, c in enumerate(cox_haz):
        plt.step(c.x, c.y, where="post",label='Cox')
    plt.legend()
        