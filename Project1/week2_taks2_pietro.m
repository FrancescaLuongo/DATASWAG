%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% TRAINING & TEST ERRORS %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER
% features selection
step = 9; % step = 1 per averle tutte
featuresSelection = 1:step:nFeatures;
nSelectedFeatures = size(featuresSelection,2);

% set1 & set2 selection
cvp = cvpartition(nObservations,'Holdout',0.1);
idxSet1 = training(cvp);
idxSet2 = test(cvp);

dataSet1 = trainData(idxSet1,featuresSelection);
dataLabel1 = trainLabels(idxSet1);
dataSet2 = trainData(idxSet2,featuresSelection);
dataLabel2 = trainLabels(idxSet2);

%% TRAIN ON SET 1

% INIZIALIZZA
mClass = modelClassificationClass();
mClass = mClass.setTrainData(dataSet1,dataLabel1);

%modelClass1 = modelClass1.setPriorProbability('uniform');

% ALLENA
mClass = mClass.train();
modelTypes = mClass.getModelTypes();

%% ERRORS:
% TRAIN ERROR:
trainRes = mClass.structuredResults(dataSet1,dataLabel1);
testRes = mClass.structuredResults(dataSet2,dataLabel2);


%% PRIOR P UNIFORM

% INIZIALIZZA
mClassUniform = modelClassificationClass();
mClassUniform = mClassUniform.setTrainData(dataSet1,dataLabel1);

% SET PRIOR PROBABILITY
mClassUniform = mClassUniform.setPriorProbability('uniform');

% ALLENA
mClassUniform = mClassUniform.train();
modelTypes = mClassUniform.getModelTypes();

%% ERRORS:
% TRAIN ERROR:
trainResUniform = mClassUniform.structuredResults(dataSet1,dataLabel1);
testResUniform = mClassUniform.structuredResults(dataSet2,dataLabel2);


%% PRIOR P DEFINED

% INIZIALIZZA
mClassPrior = modelClassificationClass();
mClassPrior = mClassPrior.setTrainData(dataSet1,dataLabel1);

% SET PRIOR PROBABILITY
mClassPrior = mClassPrior.setPriorProbability([0.3,0.7]);

% ALLENA
mClassPrior = mClassPrior.train();
modelTypes = mClassPrior.getModelTypes();

%% ERRORS:
% TRAIN ERROR:
trainResPrior = mClassPrior.structuredResults(dataSet1,dataLabel1);
testResPrior = mClassPrior.structuredResults(dataSet2,dataLabel2);


%% AND PLOT
figure;
bar([trainResPrior.classErrors,testResPrior.classErrors]);
set(gca,'xticklabel',modelTypes');
title('class error');
legend('train error','test error');
xlabel('(a)');

%%
figure;
bar([trainResPrior.classificationErrors,testResPrior.classificationErrors]);
set(gca,'xticklabel',modelTypes');
title('classification error');
legend('train error','test error');



