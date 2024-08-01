# -*- coding: utf-8 -*-
"""
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
import seaborn as sns
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.stats import false_discovery_control

#Set classifier and hyperparameter optimization method.
args = docopt(__doc__)
classifier = args['<classifier>']
scorfunc = args['<scorfunc>']
rep_size = args['<rep_size>']
outer_size = args['<outer_size>']
inner_size = args['<inner_size>']
nperm = args['<nperm>']
# classifier = 'RF'
# scorfunc = 'f1'
# rep_size = '5'
# outer_size = '10'
# inner_size = '5'
# nperm = '1000'
print(classifier,scorfunc,rep_size,outer_size,inner_size,nperm)

#Set path.
basepath = ('binary_'+classifier+'_'+scorfunc+'_r'+rep_size+'_fo'+outer_size+'_fi'+inner_size+'/')
outpath = (basepath+'gather/')

#Set and reformat.
ycat = ['good-prognosis','remitting-course','clinical-worsening','persistent-course']
ncat = len(ycat)

#Read in the p-values for summary scores for each category - specifically, F1.
accsel = 'f1'
sump = []
for catidx in range(ncat):
    infile = (outpath+str(catidx)+'_summary_scores_OneP.csv')
    inmat = pd.read_csv(infile,index_col=0,header=None).loc[accsel,1]
    sump.append(inmat)
sump = pd.Series(sump,index=ycat)
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
fdrp_vec = false_discovery_control(sump_vec,method='bh')
fdrp = pd.DataFrame(fdrp_vec.reshape(sump.shape),index=sump.index,columns=sump.columns)
outfile = (outpath+'all_feature_OneP.csv')
sump.to_csv(outfile)
outfile = (outpath+'all_feature_OneP_FDR.csv')
fdrp.to_csv(outfile)
print('Saved.')
