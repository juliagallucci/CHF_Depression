# -*- coding: utf-8 -*-
"""
Usage: 
    heartseek_score_hyperopt_binary_perm.py <curr_cat> <classifier> <scorfunc> <rep_size> <outer_size> <inner_size> <replab> <outerlab> <nperm> <permidx>
    
Arguments:
    
    <curr_cat> Current positive category
    <classifier> Classifier
    <scorfunc> Scoring function for hyperopt if doing it
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds
    <inner_size> Number of inner folds
    <replab> Current repetition
    <outerlab> Current fold
    <nperm> Number of permutations
    <permidx> Current permutation index

"""

import time, os, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from hpsklearn import HyperoptEstimator, random_forest_classifier
from hyperopt import fmin, tpe, atpe, hp, anneal, Trials
from docopt import docopt

#Set classifier and hyperparameter optimization method.
args = docopt(__doc__)
curr_cat = args['<curr_cat>']
classifier = args['<classifier>']
scorfunc = args['<scorfunc>']
rep_size = args['<rep_size>']
outer_size = args['<outer_size>']
inner_size = args['<inner_size>']
replab = args['<replab>']
outerlab = args['<outerlab>']
nperm = args['<nperm>']
permidx = args['<permidx>']
# curr_cat = '0'
# classifier = 'RF'
# scorfunc = 'f1'
# rep_size = '5'
# outer_size = '10'
# inner_size = '5'
# replab = '1'
# outerlab = '1'
# nperm = '1000'
# permidx = '1'
print(curr_cat,classifier,scorfunc,rep_size,outer_size,inner_size,replab,outerlab,nperm,permidx)

#Reformat and set full seed.
fullseed = 12345
nevals = 500
ccat = int(curr_cat)
nrep = int(rep_size)
inner_k = int(inner_size)
outer_k = int(outer_size)

#Set constant parameters.
if classifier == 'RF':

    #Number of trees. As high as you can go is ok. 
    #Talked about in class.
    ntrees = 500

#Read in the input matrix.
infile = 'heartseek_XY_2.csv'
inmat = pd.read_csv(infile)

#Produce labels.
ylabs = ['tr_labels','tr_idx']
xlabs = [x for x in inmat.columns if x not in ylabs]
nfeat = len(xlabs)
nsample = inmat.shape[0]
ycat = ['good-prognosis','remitting-course','clinical-worsening','persistent-course']
ncat = len(ycat)

#Divide.
data_X = inmat.loc[:,xlabs]
data_Y = (inmat.loc[:,'tr_idx']==ccat).astype(int)
data_Y.name = ycat[ccat]

#Read outer CV test fold indices.
infile = ('heartseek_cv_r'+rep_size+'_o'+outer_size+'_i'+inner_size+'.csv')
outercv_test = pd.read_csv(infile,header=None).values

#Check to see there are no repeats.
# column_tuples = [tuple(column) for column in outercv_test.T]
# unique_columns = set(column_tuples)
# has_duplicates = len(column_tuples) != len(unique_columns)
# print('Has duplicates:',has_duplicates)

#Set number of test and train samples.
ntest = int(np.ceil(nsample/outer_k))
ntrain = nsample - ntest

#Define repetition and outer fold labels.
rk_labs = []
for ridx in range(nrep):
    for outidx in range(outer_k):
        rk_labs.append('r'+str(ridx+1)+'_f'+str(outidx+1))
nrk = len(rk_labs)

#Define all CV repetition seeds from the full seed.
np.random.seed(fullseed)
repcv_list = np.random.randint(1,12345,nrep).tolist()

#Go through each repetition.
start1 = time.time()

#Set current CV repetition and outer CV iteration from the labels.
ridx = int(replab) - 1
outidx = int(outerlab) - 1

#Set the seed for this CV repetition.
repseed = repcv_list[ridx]

#Define outer CV seeds from the CV repetition seed.
np.random.seed(repseed)
outcv_list = np.random.randint(1,12345,outer_k).tolist()

#Extract outer CV indices for this repetition.
outercollect = outercv_test[:,ridx]

#Go through outer CV iterations.
rk_lab = 'r'+str(ridx+1)+'_f'+str(outidx+1)
print(rk_lab)

#Read in the hyperparameters from the fitted model on the true data.
inpath = ('binary_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/')
savelab = 'hyper'
infile = (inpath+curr_cat+'_binary_'+rk_lab+'.h5')
savestore = pd.HDFStore(infile,'r')
savekey = ('/'+savelab+'_'+rk_lab)
best_min = savestore.select(savekey)
savestore.close()

#Set the seed for this outer CV iteration.
outerseed = outcv_list[outidx]

#Extract indices.
train_index = (np.where(outercollect!=(outidx+1))[0]).tolist()
test_index = (np.where(outercollect==(outidx+1))[0]).tolist()

#Generate permutation indices for all the permutations.
permseed = 12345
random.seed(permseed)
lorig = tuple(range(nsample))
lnum = list(range(nsample))
pset = set()
pset.add(tuple(lnum))
while len(pset) < (int(nperm)+1):
    random.shuffle(lnum)
    pset.add(tuple(lnum))
pset = list(pset)  
pset.remove(lorig)

#Define permutation set for the current permutation.
oneset = list(pset[(int(permidx)-1)])

#Extract permuted Y and relabel.
perm_Y = data_Y.iloc[oneset]
perm_Y.index = data_Y.index

#Extract train and test X and Y.
X_train, X_test = data_X.iloc[train_index,:], data_X.iloc[test_index,:]
Y_train, Y_test = perm_Y.iloc[train_index], perm_Y.iloc[test_index]

#Produce actual classifier with the best hyperparameters.
chyper = RandomForestClassifier(random_state=outerseed,
                                n_estimators=ntrees,
                                criterion=best_min['criterion'],
                                class_weight=best_min['class_weight'],
                                max_depth=best_min['max_depth'],
                                max_features=best_min['max_features'],
                                min_samples_leaf=best_min['min_samples_leaf'],
                                min_samples_split=best_min['min_samples_split'])

#Fit.
chyper.fit(X_train,Y_train)

#Generate predicted labels, predicted label probabilities.
Y_test_predict = pd.Series(chyper.predict(X_test),index=Y_test.index)
Y_test_proba = pd.DataFrame(chyper.predict_proba(X_test),index=Y_test.index)

#Confirm that they are the same without permutation.
# inpath = ('binary_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/')
# savelab = 'y_predict'
# infile = (inpath+curr_cat+'_binary_'+rk_lab+'.h5')
# savestore = pd.HDFStore(infile)
# savekey = ('/'+savelab+'_'+rk_lab)
# y_predict_confirm = savestore.select(savekey)
# savestore.close()
# savelab = 'y_proba'
# infile = (inpath+curr_cat+'_binary_'+rk_lab+'.h5')
# savestore = pd.HDFStore(infile)
# savekey = ('/'+savelab+'_'+rk_lab)
# y_proba_confirm = savestore.select(savekey)
# savestore.close()
# np.sum(y_predict_confirm == Y_test_predict)
# np.sum(y_proba_confirm == Y_test_proba,axis=0)

#Generate impurity-based feature importance.
if classifier == 'RF':
    featimp = pd.Series(chyper.feature_importances_,index=X_train.columns)

#Set output path.
outpath = ('binary_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/perm/')
os.makedirs(outpath,exist_ok=True)

#Save everything into h5 file to store.
outfile = (outpath+curr_cat+'_binary_'+rk_lab+'_p'+permidx+'.h5')
savelist = [Y_test,Y_test_predict,Y_test_proba,featimp]
savelabs = ['y_true','y_predict','y_proba','featimp']
nsave = len(savelist)
for saidx in range(nsave):
    savestore = pd.HDFStore(outfile)
    savemat = savelist[saidx]
    savelab = savelabs[saidx]
    savekey = ('/'+savelab+'_'+rk_lab)
    savestore.put(savekey,savemat)
    # atest = savestore.select(savekey)
    savestore.close()

#Display time.
end1 = time.time()
print('Permutation done:',end1-start1)
