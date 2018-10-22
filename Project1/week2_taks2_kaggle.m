%% LOADING DATA & VARS

loadingInVars_pietro;

%%%%%%%%% TRAINING & TEST ERRORS %%%%%%%%%%%
%% SETTA I DATI PER IL TRAINER CON RANDOM FEATURES
% features selection

% per ora le features migliori sembrerebbero essere quelle del
% diagquadratic_02_70F.mat e quelle del linear_diaglinear_diagquadratic_01
load('./goodRandomFeatures/linear_03_94F.mat');
nSelectedFeatures = sum(selectedFeatures);

%% TRAIN ON TRAIN DATA

% INIZIALIZZA
classifierModel = modelClassificationClass();
classifierModel = classifierModel.setTrainData(trainData(idxTraining,selectedFeatures),trainLabels(idxTraining));

% ALLENA
classifierModel = classifierModel.train();
modelTypes = classifierModel.getModelTypes();

%% ERRORS:
% TEST ERROR sui test label stimati (magari non sono veri)
testRes = classifierModel.structuredResults(testData(:,selectedFeatures),testLabels);


%% AND PLOT
figure;
bar(testRes.classErrors);
set(gca,'xticklabel',modelTypes');
title('class error');


%% SAVE PREDICITON

labelToCSV(testRes.predictions(:,3),'prediction_linear_03_94F.csv','./predictions/')
