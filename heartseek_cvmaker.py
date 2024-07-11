# -*- coding: utf-8 -*-
"""

Take in cross-validation repetition number and outer fold number then output
test fold subject indices for each outer fold and repetition.

Usage: 
    heartseek_cvmaker.py <rep_size> <outer_size>
    
Arguments:
    
    <rep_size> Number of repetitions
    <outer_size> Number of outer folds

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from docopt import docopt

#Set CV parameters.
args = docopt(__doc__)
rep_size = args['<rep_size>']
outer_size = args['<outer_size>']
print(rep_size,outer_size)

#Set full seed and retrieve numeric arguments.
fullseed = 12345
nrep = int(rep_size)
outer_k = int(outer_size)

#Read in the input matrix and extract dimension.
infile = 'heartseek_XY.csv'
inmat = pd.read_csv(infile)
nsample = inmat.shape[0]

#Divide input matrix into X features and Y label.
ylabs = ['tr_labels','tr_idx']
xlabs = [x for x in inmat.columns if x not in ylabs]
data_X = inmat.loc[:,xlabs]
data_Y = inmat.loc[:,'tr_idx']

#Set up outer CV train-test indices for every repetition. Specifically, give
#a label to test subject indices for each outer CV iteration for each repetition.
outercv_test = []

#Set the seed to initialize. While we don't have all the repetition indices,
#run the loop.
np.random.seed(fullseed)
while len(outercv_test) < nrep:

    #Sample to get a new seed to input.
    repseed = np.random.randint(1,12345)

    #Set up outer CV generator from this seed.
    outer_kf = StratifiedKFold(n_splits=outer_k,shuffle=True,random_state=repseed)

    #Pull out the subject indices for each test fold.
    allidx = np.zeros((nsample))
    for outidx, (_,test_idx) in enumerate(outer_kf.split(data_X,data_Y)):
        allidx[test_idx] = outidx + 1
    
    #If this fold arrangement doesn't exist in the list yet, append it for the
    #repetition.
    idx_exist = any(np.array_equal(allidx,arr) for arr in outercv_test)
    if not idx_exist:
        outercv_test.append(allidx)
    else:
        print('Indices exist.')

#Convert to array.
outercv_test = np.array(outercv_test).T

#Save these test fold subject indices for each repetition.
outfile = ('heartseek_cv_r'+rep_size+'_o'+outer_size+'.csv')
pd.DataFrame(outercv_test).to_csv(outfile,header=None,index=None)
