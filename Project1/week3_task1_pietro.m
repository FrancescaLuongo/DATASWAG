%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% per la riproducibilità
seed = 45;
rng(seed);

orderedFeatures = rankfeat(trainData,trainLabels,'fisher');

startN = 1;
stopN = 1000;
k_fold = 10;

classErrors = [];

tic();

for nDF = startN:stopN %nDF = number of discriminant features
    selectedFeatures = orderedFeatures(1:nDF);
    
    [cvErrors,modelTypes] = ...
        kcvClassifier(trainData(:,selectedFeatures),trainLabels,'kfold',k_fold,'modelTypes',{'diagquadratic'});
        
    classErrors = [classErrors,cvErrors.classErrorsMean];
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






