from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,make_scorer
from sklearn.metrics import auc,precision_score,recall_score, f1_score, roc_curve,accuracy_score

import matplotlib.pyplot as plt


def tune_clf(clf,Xtr,ytr,Xtst,ytst,grid,scorers,score='recall_score'):
    
    #skf = StratifiedKFold(n_splits=5)
    clf_cv = RandomizedSearchCV(clf, grid, cv=5, scoring=scorers, refit=score,return_train_score=True)
    clf_cv.fit(Xtr,ytr)
    print("Tuned Parameters: {}".format(clf_cv.best_params_)) 
    print('')
    
    # By default the search returns a classifier already fit with the optimal hyperparameters (refit=True)
    # so there's no need to refit
    ypred = clf_cv.predict(Xtst)
    ypred_prob=clf_cv.predict_proba(Xtst)
    print_scores(ytst,ypred)
   
    return clf_cv


def print_scores(ytst,ypred):
    
    print('Accuracy Score : ' + str(accuracy_score(ytst,ypred)))
    print('Precision Score : ' + str(precision_score(ytst,ypred)))
    print('Recall Score : ' + str(recall_score(ytst,ypred)))
    print('F1 Score : ' + str(f1_score(ytst,ypred)))
    print('Confusion matrix: \n', confusion_matrix(ytst, ypred))
    
    
def plot_roc(ytest,ypred_pro):
        
    fpr, tpr, _ = roc_curve(ytest, ypred_pro[:,1])
    roc_auc = auc(fpr, tpr)
    plt.subplots(1,1,figsize=(10,5))
    plt.plot(fpr,tpr, lw=2,)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title('Receiver operating characteristic',fontsize=16)
