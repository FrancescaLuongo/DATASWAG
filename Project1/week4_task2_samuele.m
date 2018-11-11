%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%

%% Sequential forward selection
%TODO ADD ERROR FUNCTION 

tic();
fun = @(xT,yT,xt,yt) length(yt)*(findCEByLabels(yt,predict(fitcdiscr(xT,yT,...
    'discrimtype', 'diaglinear'), xt)));

opt = statset('Display','iter','MaxIter',100);

kfold=3;
CrossValidationPartition = cvpartition(nObservations,'KFold',kfold);

[sel,hst] = sequentialfs(fun,trainData,trainLabels,...
    'cv',CrossValidationPartition,'options',opt);

hst.Crit(end)
toc();