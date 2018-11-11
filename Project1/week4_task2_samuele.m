%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%

%% Sequential forward selection
%TODO ADD ERROR FUNCTION 

fun = @(xT,yT,xt,yt) length(yt)*(immse(yt,predict(fitcdiscr(xT,yT,...
    'discrimtype', 'diaglinear'), xt)));

opt = statset('Display','iter','MaxIter',100);

kfold=10;
CrossValidationPartition = cvpartition(nObservations,'KFold',kfold);

[sel,hst] = sequentialfs(fun,trainData,trainLabels,...
    'cv',CrossValidationPartition,'options',opt);

hst.Crit(end)