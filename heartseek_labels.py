"""

Extract the baseline and follow-up depression scores, cluster baseline and follow-up 
scores using GMM where the best cluster number is determined using silhouette score, 
construct trajectories based on baseline and follow-up clusters, output trajectory 
labels and features of interest. 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib import cm

#Function to plot density plots with cluster coloring.
def plot_cluster_density(x,centroids,labels,title):
    '''
    Reads in datapoints, centroids, data point labels, and the title. Plots the
    data points as density plots with different label colors and centroids as 
    vertical lines with the title.
    Arguments:
        x: Numpy array, values n_datapoints x 1
        centroids: Numpy array, centroids n_centroids x 1
        labels: Numpy array, labels n_datapoints
        title: string, plot title
    '''

    #Format.
    plt.figure(figsize=(5, 5))
    cmap = cm.get_cmap('viridis')

    #For each cluster.
    for i in range(len(centroids)):

        #Extract data with the current cluster.
        cluster_data = x[labels == i]

        #Plot density plot with a specific coloring for the cluster.
        plt.hist(cluster_data,density=True,bins=np.arange(-1,50,1),alpha=0.5,label=f'Cluster {i+1}',color=cmap(float(i)/len(centroids)))

        #Plot centroid line.
        plt.axvline(x=centroids[i],color='red',linestyle='--',linewidth=2)
    
    #Format.
    plt.xlim(-1,50)
    plt.title(title,fontsize=20)
    plt.xlabel('BDI Scores',fontsize=20)
    plt.ylabel('Density',fontsize=20)
    plt.legend()
    plt.show()

#Set random state for random processes and number of repetitions for the GMM.
randstate = 12345
nrep = 500

#Set labels of interest, baseline and follow-up.
cont_y = ['bdi_bl','bdi_fup_tot']

#Set discrete variables of interest.
disc_x = ['gender','education','marital','employment','history_depression',
          'lvf','ischemic_chf','diabetes','copd','stroke', 'renal_disease',
          'smoking','bypass_surgery', 'infarction','nyha_class','ef_cat']

#Set continuous variables of interest.
cont_x = ['age','month_incom','lecl_bl','mspss_so_bl','mspss_family_bl',
          'mspss_friends_bl','eqvas_bl','bdq_bl']

#Read input file containing all the variables.
infile = 'heart_failure_data_2.csv'
fullmat = pd.read_csv(infile)

#Isolate variables of interest, filter for those who have complete scores.
wantmat = fullmat.loc[:,cont_y+disc_x+cont_x]
wantmat = wantmat.dropna(axis=0)

#Extract baseline and follow-up scores for clustering.
bl_scores = wantmat['bdi_bl']
fup_scores = wantmat['bdi_fup_tot']

#Do GMM for 2 to 12 clusters and extract silhouette score.
klist = list(range(2,12+1))
klabs = ['k'+str(x) for x in klist]
nk = len(klist)
score_collect = pd.DataFrame(np.zeros((2,nk)),index=['Baseline','Follow-Up'],
                                              columns=klabs)
for kidx in range(nk):

    #Set number of clusters.
    kval = klist[kidx]
    klab = klabs[kidx]
    print(klab)

    #For the baseline scores, conduct GMM with that number of clusters and
    #retrieve the silhouette score from the data and the labels.
    cX = bl_scores.values.reshape(-1,1)
    gmm = GMM(n_components=kval,n_init=nrep,max_iter=1000,random_state=randstate)
    labels = gmm.fit(cX).predict(cX)
    score_collect.loc['Baseline',klab] = silhouette_score(cX,labels)

    #For the follow-up scores, repeat.
    cX = fup_scores.values.reshape(-1,1)
    gmm = GMM(n_components=kval,n_init=nrep,max_iter=1000,random_state=randstate)
    labels = gmm.fit(cX).predict(cX)
    score_collect.loc['Follow-Up',klab] = silhouette_score(cX,labels)

#Plot silhouette scores for each number of clusters at baseline and follow-up.
plt.plot(klist,score_collect.iloc[0,:], 'kx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette Score')
plt.title('BASELINE BDI: Silhouette analysis For Optimal K')
plt.xticks(klist)
plt.show()
plt.plot(klist,score_collect.iloc[1,:], 'kx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette Score')
plt.title('FOLLOW-UP BDI: Silhouette analysis For Optimal K')
plt.xticks(klist)
plt.show()

#Retrieve the number of clusters at baseline and follow-up with maximum
#silhouette score values.
ymax1 = score_collect.iloc[0,:].max()
ymax2 = score_collect.iloc[1,:].max()
ymax1_x = klabs.index(score_collect.iloc[0,:].idxmax())
ymax2_x = klabs.index(score_collect.iloc[1,:].idxmax())
bl_k = klist[ymax1_x]
fup_k = klist[ymax2_x]

#Do GMM for optimal number of clusters previously retrieved.
cX = bl_scores.values.reshape(-1,1)
gmm_bl = GMM(n_components=bl_k,n_init=nrep,max_iter=1000,random_state=randstate)
gmm_bl.fit(cX)
gmm_bl_labels = gmm_bl.predict(cX)
cX = fup_scores.values.reshape(-1,1)
gmm_fup = GMM(n_components=fup_k,n_init=nrep,max_iter=1000,random_state=randstate)
gmm_fup.fit(cX)
gmm_fup_labels = gmm_fup.predict(cX)

#Plot density plots colored by cluster labels with centroids plotted as vertical 
#lines to confirm that the clustering is visually reasonable.
cX = bl_scores.values.reshape(-1,1)
centX = gmm_bl.means_
labX = gmm_bl_labels
plot_cluster_density(cX,centX,labX,'BASELINE BDI')
cX = fup_scores.values.reshape(-1,1)
centX = gmm_fup.means_
labX = gmm_fup_labels
plot_cluster_density(cX,centX,labX,'FOLLOW-UP BDI')

#Plot labels against the values in order to annotate the labels.
plt.scatter(bl_scores,gmm_bl_labels,c=gmm_bl_labels,cmap='viridis')
plt.xlabel('Baseline BDI Scores')
plt.ylabel('Cluster Labels')
plt.colorbar(label='Cluster')
plt.show()
plt.scatter(fup_scores,gmm_fup_labels,c=gmm_fup_labels,cmap='viridis')
plt.xlabel('Follow-Up BDI Scores')
plt.ylabel('Cluster Labels')
plt.colorbar(label='Cluster')
plt.show()

#Annotate the labels based on the values and map annotations to the labels.
label_mapping = {
    1: 'minimal/mild',
    0: 'moderate/severe',
}
bl_labels = pd.Series(gmm_bl_labels).map(label_mapping)
label_mapping = {
    1: 'minimal/mild',
    0: 'moderate',
    2: 'severe'
}
fup_labels = pd.Series(gmm_fup_labels).map(label_mapping)

#Generate the raw trajectories by concatenating baseline and follow-up labels.
tr_raw = bl_labels.str.cat(fup_labels,sep='-')

#View counts of the types of raw trajectories.
print(tr_raw.value_counts())

#Combine raw trajectories into more meaningful trajectories and map the new
#trajectory labels to the raw trajectory labels.
label_mapping = {
    'moderate/severe-moderate': 'persistent-course',
    'moderate/severe-severe': 'persistent-course',
    'moderate/severe-minimal/mild': 'remitting-course',
    'minimal/mild-minimal/mild': 'good-prognosis',
    'minimal/mild-moderate': 'clinical-worsening',
    'minimal/mild-severe': 'clinical-worsening'
}
tr_labels = tr_raw.map(label_mapping)

#View counts of the types of new trajectories.
print(tr_labels.value_counts())

#Map trajectories to indices for input into the classifier later.
label_mapping = {
   'good-prognosis' : 0,
   'remitting-course' : 1,
   'clinical-worsening' : 2,
   'persistent-course' : 3,
}
tr_idx = tr_labels.map(label_mapping)

#One-hot encode discrete variables and drop first column of each to input into
#the classifier later.
disc_onehot = pd.get_dummies(wantmat[disc_x].astype(int).astype(str),drop_first=True)

#Generate full matrix containing trajectory labels, trajectory indices, one-hot
#encoded discrete feature variables, and continuous feature variables.
heartseek_X = pd.concat([wantmat[cont_x],disc_onehot],axis=1)
print(heartseek_X.columns)
heartseek_Y = pd.DataFrame([tr_labels,tr_idx],index=['tr_labels','tr_idx']).T
heartseek_X.reset_index(drop=True,inplace=True)
heartseek_Y.reset_index(drop=True,inplace=True)
heartseek_XY = pd.concat([heartseek_Y,heartseek_X],axis=1)

#Save the full matrix.
outfile = 'heartseek_XY.csv'
heartseek_XY.to_csv(outfile,index=False)
