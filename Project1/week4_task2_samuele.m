%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%

%% Sequential forward selection
%TODO ADD ERROR FUNCTION 

tic();
fun = @(xT,yT,xt,yt) length(yt)*(findCEByLabels(yt,predict(fitcdiscr(xT,yT,...
    'discrimtype', 'linear'), xt)));

opt = statset('Display','iter','MaxIter',100);

kfold=10;
CrossValidationPartition = cvpartition(nObservations,'KFold',kfold);

[sel,hst] = sequentialfs(fun,trainData,trainLabels,...
    'cv',CrossValidationPartition,'options',opt);

hst.Crit(end)
toc();

%% Training model with selected features

trainDataFfs = trainData(:,sel);

ncol = size(ans,2);
x = randperm(size(trainData,2),ncol);
randomColumns = trainData(:,x);

modelFfsDiaglinear = kcvClassifier(trainDataFfs, trainLabels, 'modelTypes', {'diagquadratic'});
modelRandomColDiaglinear = kcvClassifier(randomColumns, trainLabels, 'modelTypes', {'diagquadratic'});

%% FFS IN KCV

seed = 45;
rng(seed);
k_fold = 10;
selectedModels = {'diagquadratic'};

classErrorsPrior = [];

tic();

[cvErrors,modelTypes] = ...
    ffs_kcvClassifier(trainData,trainLabels,...
    'kfold',k_fold,'modelTypes',selectedModels,'priorProbability',[0.3,0.7]);

classErrorsPrior = [cvErrors.classErrorsMean];

toc();

