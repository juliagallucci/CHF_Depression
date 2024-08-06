# -*- coding: utf-8 -*-
"""

Take in features and trajectory labels and conduct one specific iteration of a 
repeated nested stratified k-fold cross validation for binary one-vs-all classification 
using the selected RF classifier with hyperparameters previously optimized for the 
performance metric of choice. Then with the fitted model, generate SHAP values
to characterize feature relationships.

Usage: 
    heartseek_score_hyperopt_binary_shap.py <curr_cat> <classifier> <scorfunc> <rep_size> <outer_size> <inner_size> <replab> <outerlab>
    
Arguments:
    
    <curr_cat> Current positive category
    <classifier> Classifier
    <scorfunc> Scoring function for hyperopt if doing it
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds
    <inner_size> Number of inner folds
    <replab> Current repetition
    <outerlab> Current fold

"""

import time, os, random, shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

#Set constant parameters for RF.
if classifier == 'RF':

    #Number of trees. 
    ntrees = 500

#Read outer CV test fold indices for each repetition.
infile = ('heartseek_cv_r'+rep_size+'_o'+outer_size+'_i'+inner_size+'.csv')
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

#Extract outer CV indices for this repetition.
outercollect = outercv_test[:,ridx]

#Label the current iteration.
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

#Set the seed for this outer CV iteration for use in random processes.
outerseed = outcv_list[outidx]

#Extract train and test subject indices.
train_index = (np.where(outercollect!=(outidx+1))[0]).tolist()
test_index = (np.where(outercollect==(outidx+1))[0]).tolist()

#Extract train and test X and Y.
X_train, X_test = data_X.iloc[train_index,:], data_X.iloc[test_index,:]
Y_train, Y_test = data_Y.iloc[train_index], data_Y.iloc[test_index]

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

#Generate one SHAP value per class, subject, and feature.
explainer = shap.Explainer(chyper)
shvals = explainer(X_train)

#Format. Extract for each class.
shvals_p = pd.DataFrame(shvals.values[:,:,1],index=X_train.index,columns=xlabs)
shvals_n = pd.DataFrame(shvals.values[:,:,0],index=X_train.index,columns=xlabs)

#Set output path.
outpath = ('binary_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/')
os.makedirs(outpath,exist_ok=True)

#Save everything into h5 file to store.
outfile = (outpath+curr_cat+'_binary_'+rk_lab+'.h5')
savelist = [shvals_p,shvals_n]
savelabs = ['shvals_p','shvals_n']
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
print('Feature scoring done:',end1-start1)
