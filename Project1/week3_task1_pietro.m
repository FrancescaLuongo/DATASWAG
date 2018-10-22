%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% per la riproducibilità
seed = 45;
rng(seed);

orderedFeatures = rankfeat(trainData,trainLabels,'fisher');

startN = 1;
stopN = 300;
k_fold = 10;
cvp = cvpartition(nObservations,'kfold',k_fold);

classErrors = [];

tic();

for nDF = startN:stopN %nDF = number of discriminant features
    selectedFeatures = orderedFeatures(1:nDF);
    
    [~,cvClassErrors,~,modelTypes] = ...
        kcvClassifier(trainData(:,selectedFeatures),trainLabels,k_fold,{'linear'});
        
    classErrors = [classErrors,cvClassErrors];
end
toc();

%%
x = startN:stopN;

figure;
hold on;

for i = 1:length(modelTypes)
    plot(x,classErrors(i,:));
end
legend(modelTypes);






