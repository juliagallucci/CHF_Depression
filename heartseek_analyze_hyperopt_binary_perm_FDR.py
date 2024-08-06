# -*- coding: utf-8 -*-
"""

For the p-values from the permuted repeated nested stratified k-fold cross validation for binary 
one-vs-all classification using the selected RF classifier 
with hyperparameters previously optimized for the performance metric of choice,
FDR correct across the summary score p-values and the feature importance p-values.

Usage: 
    heartseek_analyze_hyperopt_binary_perm_FDR.py <classifier> <scorfunc> <rep_size> <outer_size> <inner_size> <nperm>
    
Arguments:

    <classifier> Classifier
    <scorfunc> Scoring function for hyperopt if doing it
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds
    <inner_size> Number of inner folds
    <nperm> Number of permutations

"""

import os
import numpy as np
import pandas as pd
from docopt import docopt
from scipy.stats import false_discovery_control

#Set classifier, metric to optimize hyperparameters, CV parameters, and number 
#of permutations.
args = docopt(__doc__)
classifier = args['<classifier>']
scorfunc = args['<scorfunc>']
rep_size = args['<rep_size>']
outer_size = args['<outer_size>']
inner_size = args['<inner_size>']
nperm = args['<nperm>']
print(classifier,scorfunc,rep_size,outer_size,inner_size,nperm)

#Set base path and output path.
basepath = ('binary_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/')
outpath = (basepath+'gather/')

#Produce labels.
ycat = ['good-prognosis','remitting-course','clinical-worsening','persistent-course']
ncat = len(ycat)

#Read in the p-values for the summary score of interest for each category. Specifically,
#we chose F1 because this was optimized for. Similar p-values can be found for
#other summary metrics. 
accsel = 'f1'
sump = []
for catidx in range(ncat):
    infile = (outpath+str(catidx)+'_summary_scores_OneP.csv')
    inmat = pd.read_csv(infile,index_col=0,header=None).loc[accsel,1]
    sump.append(inmat)
sump = pd.Series(sump,index=ycat)

#FDR correct across summary metrics and save both the original p-values and the
#corrected p-values.
fdrp = pd.Series(false_discovery_control(sump,method='bh'),index=ycat)
fdrout = pd.concat((sump,fdrp),axis=1)
fdrout.columns = [accsel+'_P',accsel+'_P_FDR']
outfile = (outpath+'all_summary_scores_OneP_FDR.csv')
fdrout.to_csv(outfile)

#Read in the p-values for feature importance for each category.
sump = []
for catidx in range(ncat):
    infile = (outpath+str(catidx)+'_feature_OneP.csv')
    inmat = pd.read_csv(infile,index_col=0,header=None)
    sump.append(inmat)
sump = pd.concat(sump,axis=1)
sump.columns = ycat
sump_vec = sump.values.flatten()

#FDR correct across feature importances and save both the original p-values and the
#corrected p-values.
fdrp_vec = false_discovery_control(sump_vec,method='bh')
fdrp = pd.DataFrame(fdrp_vec.reshape(sump.shape),index=sump.index,columns=sump.columns)
outfile = (outpath+'all_feature_OneP.csv')
sump.to_csv(outfile)
outfile = (outpath+'all_feature_OneP_FDR.csv')
fdrp.to_csv(outfile)
print('Saved.')
