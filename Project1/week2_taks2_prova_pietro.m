%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% TRAINING & TEST ERRORS %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER CON RANDOM FEATURES
% features selection

% elegant code per filtrare a cazzo 
perc = 4;
featuresSelection = (randi([0 1000],1,nFeatures)/10)<=perc;
nSelectedFeatures = sum(featuresSelection);

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
modelClass1 = modelClassificationClass();
modelClass1 = modelClass1.setTrainData(dataSet1,dataLabel1);

%modelClass1 = modelClass1.setPriorProbability('uniform');

% ALLENA
modelClass1 = modelClass1.train();
modelTypes = modelClass1.getModelTypes();

%% ERRORS:
% TRAIN ERROR:
trainRes = modelClass1.structuredResults(dataSet1,dataLabel1);
testRes = modelClass1.structuredResults(dataSet2,dataLabel2);


%% AND PLOT
figure;
bar([trainRes.classErrors,testRes.classErrors]);
set(gca,'xticklabel',modelTypes');
title('class error');
legend('train error','test error');

% figure;
% bar([trainRes.classificationErrors,testRes.classificationErrors]);
% set(gca,'xticklabel',modelTypes');
% title('classification error');
% legend('train error','test error');

%% PREPARE IN CASE OF GOOD FEATURES
a = [1:nFeatures];
selectedFeatures = a(featuresSelection);
idxTraining = idxSet1;
%%
%save('./goodRandomFeatures/linear_03_94F.mat','selectedFeatures','idxTraining');
% 

% per ora le features migliori sembrerebbero essere quelle del
% diagquadratic_02_70F.mat e quelle del linear_diaglinear_diagquadratic_01
