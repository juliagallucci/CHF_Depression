# -*- coding: utf-8 -*-
"""

Take in features and trajectory labels and conduct repeated stratified k-fold cross 
validation for binary one-vs-all classification using a selected classifier from 
k-nearest neighbors (kNN), decision tree (DT), logistic regression (LoR), multilayer 
perceptron (MLP), random forest (RF), AdaBoost (ADA), and XGBoost (XGB) with 
default hyperparameters and extract classifier performance and feature importance.

Usage: 
    heartseek_score_defaults_binary.py <curr_cat> <classifier> <rep_size> <outer_size> 
    
Arguments:
    
    <curr_cat> Current positive category
    <classifier> Classifier
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds

"""

import time, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from docopt import docopt

#Set current positive category, classifier, and CV parameters.
args = docopt(__doc__)
curr_cat = args['<curr_cat>']
classifier = args['<classifier>']
rep_size = args['<rep_size>']
outer_size = args['<outer_size>']
print(curr_cat,classifier,rep_size,outer_size)

#Set main seed and set numeric arguments including current positive category, 
#CV repetitions and CV outer fold number.
fullseed = 12345
ccat = int(curr_cat)
nrep = int(rep_size)
outer_k = int(outer_size)

#Read in the input matrix.
infile = 'heartseek_XY.csv'
inmat = pd.read_csv(infile)

#Produce labels and extract dimensions.
ylabs = ['tr_labels','tr_idx']
xlabs = [x for x in inmat.columns if x not in ylabs]
nfeat = len(xlabs)
nsample = inmat.shape[0]
ycat = ['good-prognosis','remitting-course','clinical-worsening','persistent-course']
ncat = len(ycat)

#Divide input matrix into X features and Y label based on the current positive
#category.
data_X = inmat.loc[:,xlabs]
data_Y = (inmat.loc[:,'tr_idx']==ccat).astype(int)
data_Y.name = ycat[ccat]

#Read outer CV test fold indices for each repetition.
infile = ('heartseek_cv_r'+rep_size+'_o'+outer_size+'.csv')
outercv_test = pd.read_csv(infile,header=None).values

#Define repetition and outer fold labels.
rk_labs = []
for ridx in range(nrep):
    for outidx in range(outer_k):
        rk_labs.append('r'+str(ridx+1)+'_f'+str(outidx+1))
nrk = len(rk_labs)

#Define all CV repetition seeds from the main seed for use in random processes.
np.random.seed(fullseed)
repcv_list = np.random.randint(1,12345,nrep).tolist()

#Collect the accuracy, F1, ROC AUC, and PRC AUC binary score versions.
scorelabs = ['accuracy','balanced_accuracy',
             'f1','f1_weighted',
             'roc_auc','prc_auc']
nscores = len(scorelabs)
allscore_collect = pd.DataFrame(np.zeros((nrk,nscores)),index=rk_labs,columns=scorelabs)

#Collect the confusion matrices.
conflist = []

#Collect the feature importances.
feat_collect = pd.DataFrame(np.zeros((nrk,nfeat)),index=rk_labs,columns=xlabs)

#Go through each repetition.
for ridx in range(nrep):
    start1 = time.time()

    #Set the seed for this CV repetition for use in random processes.
    repseed = repcv_list[ridx]

    #Define outer CV seeds from the CV repetition seed.
    np.random.seed(repseed)
    outcv_list = np.random.randint(1,12345,outer_k).tolist()

    #Extract outer CV subject indices for this repetition.
    outercollect = outercv_test[:,ridx]

    #Go through outer CV iterations.
    for outidx in range(outer_k):

        #Set label for current iteration.
        rk_lab = 'r'+str(ridx+1)+'_f'+str(outidx+1)
        print(rk_lab)

        #Set the seed for this outer CV iteration for use in random processes.
        outerseed = outcv_list[outidx]

        #Extract train and test subject indices.
        train_index = (np.where(outercollect!=(outidx+1))[0]).tolist()
        test_index = (np.where(outercollect==(outidx+1))[0]).tolist()

        #Extract train and test X and Y.
        X_train, X_test = data_X.iloc[train_index,:].copy(), data_X.iloc[test_index,:].copy()
        Y_train, Y_test = data_Y.iloc[train_index].copy(), data_Y.iloc[test_index].copy()

        #kNN with scaling.
        if classifier == 'kNN':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            clf = KNeighborsClassifier()
            clf.fit(X_train,Y_train)

        #Decision tree.
        elif classifier == 'DT':
            clf = DecisionTreeClassifier(random_state=outerseed)
            clf.fit(X_train,Y_train)

        #Logistic regression with scaling, increase iterations for convergence.
        elif classifier == 'LoR':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            clf = LogisticRegression(random_state=outerseed,max_iter=1000)
            clf.fit(X_train,Y_train)

        #Multilayer perceptron with scaling, increase iterations for convergence.
        elif classifier == 'MLP':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            clf = MLPClassifier(random_state=outerseed,max_iter=1000)
            clf.fit(X_train,Y_train)

        #Random forest.
        elif classifier == 'RF':
            clf = RandomForestClassifier(random_state=outerseed)
            clf.fit(X_train,Y_train)

        #AdaBoost, change algorithm from one that will be deprecated in a future version.
        elif classifier == 'ADA':
            clf = AdaBoostClassifier(random_state=outerseed,algorithm='SAMME')
            clf.fit(X_train,Y_train)

        #XGBoost, limit cores used to save computation.
        elif classifier == 'XGB':
            clf = XGBClassifier(random_state=outerseed,n_jobs=1)
            clf.fit(X_train,Y_train)

        #Generate predicted labels and predicted label probabilities.
        y_true = Y_test
        y_predict = pd.Series(clf.predict(X_test),index=Y_test.index)
        y_proba = pd.DataFrame(clf.predict_proba(X_test),index=Y_test.index)

        #Generate impurity-based feature importance only, for now.
        if (classifier == 'DT') or (classifier == 'RF') or (classifier == 'ADA') or classifier == 'XGB':
            featimp = pd.Series(clf.feature_importances_,index=X_train.columns)
        else:
            featimp = pd.Series(np.zeros((nfeat)),index=xlabs)
        
        #Accuracy binary versions.
        acc = accuracy_score(y_true,y_predict)
        balacc = balanced_accuracy_score(y_true,y_predict)

        #F1 binary versions.
        f1 = f1_score(y_true,y_predict)
        f1_weighted = f1_score(y_true,y_predict,average='weighted')

        #ROC binary version.
        roc_auc = roc_auc_score(y_true,y_proba.iloc[:,1])

        #PRC binary version.
        prc_auc = average_precision_score(y_true,y_proba.iloc[:,1])

        #Put together scores and append.
        allscore = pd.Series([acc,balacc,
                              f1,f1_weighted,
                              roc_auc,prc_auc],
                            index=['accuracy','balanced_accuracy',
                                   'f1','f1_weighted',
                                   'roc_auc','prc_auc'])
        allscore_collect.loc[rk_lab,allscore.index] = allscore.values

        #Calculate confusion matrix and append.
        confmat = pd.DataFrame(confusion_matrix(y_true,y_predict))
        conflist.append(confmat)

        #Extract feature importance and append.
        feat_collect.loc[rk_lab,featimp.index] = featimp.values
 
        #Set output path.
        outpath = ('binary_'+classifier+'_default_r'+rep_size+'_fo'+outer_size+'/')
        os.makedirs(outpath,exist_ok=True)

        #Save everything into h5 file to store.
        outfile = (outpath+curr_cat+'_binary_default.h5')
        savelist = [y_true,y_predict,y_proba,featimp]
        savelabs = ['y_true','y_predict','y_proba','featimp']
        nsave = len(savelist)
        for saidx in range(nsave):
            savestore = pd.HDFStore(outfile)
            savemat = savelist[saidx]
            savelab = savelabs[saidx]
            savekey = ('/'+savelab+'_'+rk_lab)
            savestore.put(savekey,savemat)
            savestore.close()

    #Display time.
    end1 = time.time()
    print('Repetition done:',end1-start1)

#Average binary scores, confusion matrix, and feature importance across folds.
meanscore = allscore_collect.mean(axis=0)
meanconf = np.zeros((2,2))
for cconf in conflist:
    meanconf += cconf.values
meanconf /= nrk
meanconf = pd.DataFrame(meanconf,index=['other',data_Y.name],columns=['Other',data_Y.name])
meanfeat = feat_collect.mean(axis=0)

#Save binary scores.
outfile = (outpath+curr_cat+'_summary_scores.csv')
meanscore.to_csv(outfile)

#Save the confusion matrix.
outfile = (outpath+curr_cat+'_confusion_raw.csv')
meanconf.to_csv(outfile)

#Normalize confusion matrix such that each row gives the percentage of the true class
#that landed in each of the predicted classes and save.
perconf = meanconf / meanconf.sum(axis=1).values[:,np.newaxis]
outfile = (outpath+curr_cat+'_confusion.csv')
perconf.to_csv(outfile)
               
#Plot percentage confusion matrix and save.
plt.figure(figsize=(3.2,2.4))
sns.heatmap(perconf,annot=True,fmt='.2f',xticklabels=['other',data_Y.name],yticklabels=['other',data_Y.name],cbar=False)
plt.ylabel('Actual')
plt.xlabel('Predicted')
outfile = (outpath+curr_cat+'_confusion.jpg')
plt.savefig(outfile,bbox_inches='tight',dpi=720)
plt.close()

#Sort feature importances and save.
meanfeat = meanfeat.sort_values(ascending=True)
outfile = (outpath+curr_cat+'_feature.csv')
meanfeat.to_csv(outfile)

#Plot feature importances and save.
plt.barh(meanfeat.index,meanfeat.values)
plt.xlabel('Feature Importance')
outfile = (outpath+curr_cat+'_feature.jpg')
plt.savefig(outfile,bbox_inches='tight')
plt.close()
print('Saved.')
