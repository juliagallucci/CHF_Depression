# -*- coding: utf-8 -*-
"""

Take in features and trajectory labels and, for speed from parallelization, conduct 
one specific iteration of a repeated nested stratified k-fold cross validation 
for binary one-vs-all classification using the selected RF classifier with 
hyperparameters optimized for the performance metric of choice.

Usage: 
    heartseek_score_hyperopt_binary.py <curr_cat> <classifier> <scorfunc> <rep_size> <outer_size> <inner_size> <replab> <outerlab>
    
Arguments:
    
    <curr_cat> Current positive category
    <classifier> Classifier
    <scorfunc> Scoring function performance metric for hyperopt
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds
    <inner_size> Number of inner folds
    <replab> Current repetition
    <outerlab> Current fold

"""

import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from hyperopt import fmin, tpe, hp
from docopt import docopt

#Set current positive category, classifier, metric to optimize hyperparameters, 
#CV parameters, and current CV repetition and outer fold.
args = docopt(__doc__)
curr_cat = args['<curr_cat>']
classifier = args['<classifier>']
scorfunc = args['<scorfunc>']
rep_size = args['<rep_size>']
outer_size = args['<outer_size>']
inner_size = args['<inner_size>']
replab = args['<replab>']
outerlab = args['<outerlab>']
print(curr_cat,classifier,scorfunc,rep_size,outer_size,inner_size,replab,outerlab)

#Set main seed and set numeric arguments including hyperopt evaluations, current 
#positive category, CV repetitions, CV outer fold number, and CV inner fold number.
fullseed = 12345
nevals = 500
ccat = int(curr_cat)
nrep = int(rep_size)
outer_k = int(outer_size)
inner_k = int(inner_size)

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

#Set constant parameters for RF.
if classifier == 'RF':

    #Number of trees.
    ntrees = 500

#Read outer CV test fold indices for each repetition.
infile = ('heartseek_cv_r'+rep_size+'_o'+outer_size+'.csv')
outercv_test = pd.read_csv(infile,header=None).values

#Set number of test and train samples for later use with hyperparameters.
ntest = int(np.ceil(nsample/outer_k))
ntrain = nsample - ntest

#Define repetition and outer fold labels.
rk_labs = []
for ridx in range(nrep):
    for outidx in range(outer_k):
        rk_labs.append('r'+str(ridx+1)+'_f'+str(outidx+1))
nrk = len(rk_labs)

#Define all CV repetition seeds from the main seed for use in random processes.
np.random.seed(fullseed)
repcv_list = np.random.randint(1,12345,nrep).tolist()

#Start this repetition.
start1 = time.time()

#Set current CV repetition and outer CV iteration from the labels.
ridx = int(replab) - 1
outidx = int(outerlab) - 1

#Set the seed for this CV repetition for use in random processes.
repseed = repcv_list[ridx]

#Define outer CV seeds from the CV repetition seed for use in random processes.
np.random.seed(repseed)
outcv_list = np.random.randint(1,12345,outer_k).tolist()

#Extract outer CV subject indices for this repetition.
outercollect = outercv_test[:,ridx]

#Label the current iteration.
rk_lab = 'r'+str(ridx+1)+'_f'+str(outidx+1)
print(rk_lab)

#Set the seed for this outer CV iteration for use in random processes.
outerseed = outcv_list[outidx]

#Extract train and test subject indices.
train_index = (np.where(outercollect!=(outidx+1))[0]).tolist()
test_index = (np.where(outercollect==(outidx+1))[0]).tolist()

#Extract train and test X and Y.
X_train, X_test = data_X.iloc[train_index,:], data_X.iloc[test_index,:]
Y_train, Y_test = data_Y.iloc[train_index], data_Y.iloc[test_index]

#Initialize inner CV for hyperparameter optimization.
inner_kf = StratifiedKFold(n_splits=inner_k,shuffle=True,random_state=outerseed)

#Random forest hyperparameter optimization.
if classifier == 'RF':

    #Define a RF function for scoring for hyperopt, which we want to minimize.
    def rf_cv_score(params,outerseed=outerseed,inner_kf=inner_kf,X_train=X_train,Y_train=Y_train):

        #Gets hyperparameters.
        params = {'criterion': params['criterion'],
                'class_weight': params['class_weight'],
                'max_depth': params['max_depth'], 
                'max_features': params['max_features'],
                'min_samples_leaf': params['min_samples_leaf'],
                'min_samples_split': params['min_samples_split']
                }
        
        #Use these hyperparameters with classifier.
        clf = RandomForestClassifier(random_state=outerseed,n_estimators=ntrees,**params)
    
        #Conduct inner CV and retrieve the score which we want to minimize.
        cv_score = -cross_val_score(clf,X_train,Y_train,cv=inner_kf,scoring=scorfunc).mean()
        return cv_score
    
    #Define space of hyperparameters we want to explore.
    nfeat_slice = X_train.shape[1]
    criterion_list = ['gini','entropy','log_loss']
    class_weight_list = [None,'balanced','balanced_subsample']
    max_depth_list = [None] + [int(np.round(x)) for x in (np.linspace(0.1,1.0,10) * ntrain)]
    max_depth_list = [i for i in max_depth_list if i != 0]
    max_depth_list = list(dict.fromkeys(max_depth_list))
    max_features_list = ['sqrt',1] + [int(np.round(x)) for x in (np.linspace(0.1,1.0,10) * nfeat_slice)]
    max_features_list = [i for i in max_features_list if i != 0]
    max_features_list = list(dict.fromkeys(max_features_list))
    min_samples_leaf_list = [1] + [int(np.round(x)) for x in (np.linspace(0.1,1.0,10) * ntrain)]
    min_samples_leaf_list = [i for i in min_samples_leaf_list if i != 0]
    min_samples_leaf_list = list(dict.fromkeys(min_samples_leaf_list))
    min_samples_split_list = [2] + [int(np.round(x)) for x in (np.linspace(0.1,1.0,10) * ntrain)]
    min_samples_split_list = [i for i in min_samples_split_list if i != 0]
    min_samples_split_list = list(dict.fromkeys(min_samples_split_list))
    space = {'criterion' : hp.choice('criterion',criterion_list),
            'class_weight': hp.choice('class_weight',class_weight_list),
            'max_depth': hp.choice('max_depth',max_depth_list),
            'max_features': hp.choice('max_features',max_features_list),
            'min_samples_leaf': hp.choice('min_samples_leaf',min_samples_leaf_list),
            'min_samples_split': hp.choice('min_samples_split',min_samples_split_list)
            }

    #Fit minimizer for best hyperparameters, selecting TPE algorithm.
    best_min = fmin(fn=rf_cv_score,
                space=space, 
                algo=tpe.suggest,
                max_evals=int(nevals),
                rstate=np.random.default_rng(outerseed),
                return_argmin=False)
    
    #Produce classifier with the best hyperparameters.
    chyper = RandomForestClassifier(random_state=outerseed,
                                    n_estimators=ntrees,
                                    criterion=best_min['criterion'],
                                    class_weight=best_min['class_weight'],
                                    max_depth=best_min['max_depth'],
                                    max_features=best_min['max_features'],
                                    min_samples_leaf=best_min['min_samples_leaf'],
                                    min_samples_split=best_min['min_samples_split'])

#Fit the classifier with the best hyperparameters.
chyper.fit(X_train,Y_train)

#Generate best hyperparameters.
best_hyper = pd.Series(best_min.values(),index=best_min.keys())

#Generate predicted labels and predicted label probabilities.
Y_test_predict = pd.Series(chyper.predict(X_test),index=Y_test.index)
Y_test_proba = pd.DataFrame(chyper.predict_proba(X_test),index=Y_test.index)

#Generate impurity-based feature importance only, for now.
if classifier == 'RF':
    featimp = pd.Series(chyper.feature_importances_,index=X_train.columns)

#Set output path.
outpath = ('binary_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/')
os.makedirs(outpath,exist_ok=True)

#Save everything into h5 file to store.
outfile = (outpath+curr_cat+'_binary_'+rk_lab+'.h5')
savelist = [Y_test,Y_test_predict,Y_test_proba,best_hyper,featimp]
savelabs = ['y_true','y_predict','y_proba','hyper','featimp']
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
print('Iteration done:',end1-start1)
