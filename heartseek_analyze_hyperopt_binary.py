# -*- coding: utf-8 -*-
"""

Collect iterations for a repeated nested stratified k-fold cross validation 
for binary one-vs-all classification using the selected RF classifier with 
hyperparameters optimized for the performance metric of choice and extract 
classifier performance and feature importance.

Usage: 
    heartseek_analyze_hyperopt_binary.py <curr_cat> <classifier> <scorfunc> <rep_size> <outer_size> <inner_size>
    
Arguments:
    
    <curr_cat> Current positive category
    <classifier> Classifier
    <scorfunc> Scoring function performance metric for hyperopt
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds
    <inner_size> Number of inner folds

"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from docopt import docopt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

#Set current positive category, classifier, metric to optimize hyperparameters, 
#and CV parameters.
args = docopt(__doc__)
curr_cat = args['<curr_cat>']
classifier = args['<classifier>']
scorfunc = args['<scorfunc>']
rep_size = args['<rep_size>']
outer_size = args['<outer_size>']
inner_size = args['<inner_size>']
print(curr_cat,classifier,scorfunc,rep_size,outer_size,inner_size)

#Set base path and output path.
basepath = ('binary_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/')
outpath = (basepath+'gather/')
os.makedirs(outpath,exist_ok=True)

#Set numeric arguments including current positive category, CV repetitions, 
#CV outer fold number, and CV inner fold number.
ccat = int(curr_cat)
nrep = int(rep_size)
inner_k = int(inner_size)
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

#Define repetition and outer fold labels.
rk_labs = []
for ridx in range(nrep):
    for outidx in range(outer_k):
        rk_labs.append('r'+str(ridx+1)+'_f'+str(outidx+1))
nrk = len(rk_labs)

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

#Go through each repetition and outer CV iteration.
for ridx in range(nrep):
    for outidx in range(outer_k):

        #Set label for current iteration.
        rk_lab = 'r'+str(ridx+1)+'_f'+str(outidx+1)

        #Open store.
        infile = (basepath+curr_cat+'_binary_'+rk_lab+'.h5')
        instore = pd.HDFStore(infile,'r')

        #Extract true, predicted, and predicted probability labels.
        inlab = 'y_true'
        inkey = ('/'+inlab+'_'+rk_lab)
        y_true = instore.select(inkey)
        inlab = 'y_predict'
        inkey = ('/'+inlab+'_'+rk_lab)
        y_predict = instore.select(inkey)
        inlab = 'y_proba'
        inkey = ('/'+inlab+'_'+rk_lab)
        y_proba = instore.select(inkey)

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
        inlab = 'featimp'
        inkey = ('/'+inlab+'_'+rk_lab)
        inmat = instore.select(inkey)
        feat_collect.loc[rk_lab,inmat.index] = inmat.values

        #Close store.
        instore.close()

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