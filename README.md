# CHF_Depression

1. heartseek_labels.py

Extract the baseline and follow-up depression scores, cluster baseline and follow-up 
scores using GMM where the best cluster number is determined using silhouette score, 
construct trajectories based on baseline and follow-up clusters, output trajectory 
labels and features of interest. 

2. heartseek_cvmaker.py

Take in cross-validation repetition number and outer fold number then output
test fold subject indices for each outer fold and repetition.

3. heartseek_score_defaults_binary.py

Take in features and trajectory labels and conduct repeated stratified k-fold cross 
validation for binary one-vs-all classification using a selected classifier from 
k-nearest neighbors (kNN), decision tree (DT), logistic regression (LoR), multilayer 
perceptron (MLP), random forest (RF), AdaBoost (ADA), and XGBoost (XGB) with 
default hyperparameters and extract classifier performance and feature importance.

4. heartseek_score_hyperopt_binary.py

Take in features and trajectory labels and, for speed from parallelization, conduct 
one specific iteration of a repeated nested stratified k-fold cross validation 
for binary one-vs-all classification using the selected RF classifier with 
hyperparameters optimized for the performance metric of choice.

5. heartseek_analyze_hyperopt_binary.py

Collect iterations for a repeated nested stratified k-fold cross validation 
for binary one-vs-all classification using the selected RF classifier with 
hyperparameters optimized for the performance metric of choice and extract 
classifier performance and feature importance.

6. heartseek_score_hyperopt_binary_perm.py

Take in features and trajectory labels, permute the trajectory labels, and conduct 
one specific iteration of a permuted repeated nested stratified k-fold cross 
validation for binary one-vs-all classification using the selected RF classifier with 
hyperparameters previously optimized for the performance metric of choice.

7. heartseek_analyze_hyperopt_binary_perm.py

Collect iterations for a permuted repeated nested stratified k-fold cross 
validation for binary one-vs-all classification using the selected RF classifier 
with hyperparameters previously optimized for the performance metric of choice 
and extract one-sided p-values for the classifier performance and feature importance.

8. heartseek_analyze_hyperopt_binary_perm_FDR.py

For the p-values from the permuted repeated nested stratified k-fold cross validation for binary 
one-vs-all classification using the selected RF classifier 
with hyperparameters previously optimized for the performance metric of choice,
FDR correct across the summary score p-values and the feature importance p-values.

9. heartseek_score_hyperopt_binary_shap.py 

Take in features and trajectory labels and conduct one specific iteration of a 
repeated nested stratified k-fold cross validation for binary one-vs-all classification 
using the selected RF classifier with hyperparameters previously optimized for the 
performance metric of choice. Then with the fitted model, generate SHAP values
to characterize feature relationships.

10. heartseek_analyze_hyperopt_binary_shap.py

Collect iterations for a repeated nested stratified k-fold cross 
validation for binary one-vs-all classification using the selected RF classifier 
with hyperparameters previously optimized for the performance metric of choice 
and extract SHAP values from each iteration for the positive class. Average all 
SHAP values for each subject and plot beeswarm plots based on them.

