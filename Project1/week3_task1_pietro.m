%% LOADING DATA & VARS
loadingInVars_pietro;

%%%%%%%%% LDA/QDA classifiers %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% per la riproducibilità

k_fold = 20;
modelTypes = {'linear','diagquadratic'};
startF = 1;
endF = 150;
stepF = 5;

tic();

% init kcv:
clear('kcv');
kcv = crossValidationClass();
kcv = kcv.setData(trainData,trainLabels);
kcv = kcv.setKFold(k_fold);
kcv = kcv.setModelTypes(modelTypes);
kcv = kcv.setTrainErrors(false);
kcv = kcv.setPriorProbability('empirical');

% se vogliamo un feature selection INSIDE la cv
kcv = kcv.setCV_type('fisher'); % fisher, normal, pca, ...

% se vogliamo variare dei parametri, ex: nFeatures varia il numero di
% features da prendere
kcv.parameterVariation = 'nFeatures'; % nFeatures, altro, ...
kcv = kcv.setStartStop(startF,endF,stepF);

% and run
kcv = kcv.runCV();

toc();

%% plot data:

x = startF:stepF:endF;

figure;
hold on;
res = kcv.getResultsByType('linear','classErrorsMean');
plot(x,res)
res = kcv.getResultsByType('diagquadratic','classErrorsMean');
plot(x,res)

legend(modelTypes);
xlabel({'number of features','(b)'});
ylabel('Class errors mean');





%% OLD CODE
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






