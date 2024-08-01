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

6. heartseek_analyze_hyperopt_binary_perm.py
Collect iterations for a repeated nested stratified k-fold cross-validation
for binary classification using the specified classifier and scoring function,
optimize hyperparameters, extract p-values for summary scores, and feature importance.

7. heartseek_analyze_hyperopt_binary_perm_FDR.py
Collect iterations for a repeated nested stratified k-fold cross-validation
for binary classification using the specified classifier and scoring function,
optimize hyperparameters, extract p-values for summary scores, and feature importance,
and apply false discovery rate control.

8. heartseek_score_hyperopt_binary.py
Take in features and trajectory labels and, for speed from parallelization,
conduct one specific iteration of a repeated nested stratified k-fold cross-validation
for binary one-vs-all classification using the selected classifier with hyperparameters
optimized for the performance metric of choice, while performing permutation-based
evaluations and saving predictions and feature importance.

