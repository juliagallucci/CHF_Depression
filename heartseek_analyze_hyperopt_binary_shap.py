# -*- coding: utf-8 -*-
"""

Collect iterations for a repeated nested stratified k-fold cross 
validation for binary one-vs-all classification using the selected RF classifier 
with hyperparameters previously optimized for the performance metric of choice 
and extract SHAP values from each iteration for the positive class. Average all 
SHAP values for each subject and plot beeswarm plots based on them.

Usage: 
    heartseek_analyze_hyperopt_binary_shap.py <curr_cat> <classifier> <scorfunc> <rep_size> <outer_size> <inner_size>
    
Arguments:
    
    <curr_cat> Current positive category
    <classifier> Classifier
    <scorfunc> Scoring function for hyperopt if doing it
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds
    <inner_size> Number of inner folds

"""

import os, shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

#Collect the sum of feature importances per sample for the positive class and
#the count to average later.
nsub = data_X.shape[0]
sublist = data_X.index.values
subsum = pd.DataFrame(np.zeros((nsub,nfeat)),index=sublist,columns=xlabs)
subcount = pd.Series(np.zeros((nsub)),index=sublist)

#Fill in collectors.
for ridx in range(nrep):
    for outidx in range(outer_k):
        rk_lab = 'r'+str(ridx+1)+'_f'+str(outidx+1)

        #Open store.
        infile = (basepath+curr_cat+'_binary_'+rk_lab+'.h5')
        instore = pd.HDFStore(infile,'r')

        #Extract feature importance for the positive class.
        inlab = 'shvals_p'
        inkey = ('/'+inlab+'_'+rk_lab)
        shvals_p = instore.select(inkey)
        instore.close()

        #Add features and count by subject.
        csublist = shvals_p.index
        subsum.loc[csublist,:] += shvals_p
        subcount.loc[csublist] += 1

#Divide the sum by the count to get the average.
submean = subsum.div(subcount,axis=0)

#Save the average SHAP values.
outfile = (outpath+curr_cat+'_feature_shap_1.csv')
submean.to_csv(outfile,index=False)

#Read in the impurity-based feature importance for the sorting.
infile = (outpath+curr_cat+'_feature.csv')
inmat = pd.read_csv(infile,index_col=0)
xlabs_order = inmat.index.values[::-1]

#Generate the beeswarm plot.
shap.summary_plot(submean.loc[:,xlabs_order].values,data_X.loc[:,xlabs_order],
                  show=False,sort=False,max_display=nfeat)
outfile = (outpath+curr_cat+'_beeswarm_shap_1.jpg')
plt.savefig(outfile,bbox_inches='tight',dpi=720)
plt.close()
print('Saved.')
