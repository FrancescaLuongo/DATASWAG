%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%
%% CHECK CV PARTITION

k_fold = 10;
cvp = cvpartition(nObservations,'kfold',k_fold);
 
%%
idxSetTraining = cvp.training(1);
idxSetTest = cvp.test(1);