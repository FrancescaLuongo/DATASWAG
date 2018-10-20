clc; clear all; close all;

load('trainSet.mat')
load('trainLabels.mat')
load('testSet.mat')
%addpath('usefulFunctions');

[~,~,L1] = modelClassification(trainData, trainLabels,'Linear')
[~,~,L2] = modelClassification(trainData, trainLabels,'diagLinear')
[~,~,L3] = modelClassification(trainData, trainLabels,'diagQuadratic')



plot(L2,'mo') %3rd
hold on;
plot(L1,'go') %best
hold on;
plot(L3,'co') %2nd
hold on;


%% Prior test

% classifier = fitcdiscr(trainData, trainLabels, 'DiscrimType', 'diagLinear', 'Prior', 'uniform')
% 
% 
% yhat = predict(classifier, trainData);
% 
% loss =loss(classifier, trainData, trainLabels)
% 

%% with prior

[~,~,L1] = Prior_modelClassification(trainData, trainLabels,'Linear')
[~,~,L2] = Prior_modelClassification(trainData, trainLabels,'diagLinear')
[~,~,L3] = Prior_modelClassification(trainData, trainLabels,'diagQuadratic')

plot(L2,'mo') %3rd
hold on;
plot(L1,'go') %best
hold on;
plot(L3,'co') %2nd
hold on;



%%



%% pb avec celui-ci
% fonctionne pas car: "You cannot use 'quadratic' type because one or more classes have singular covariance matrices."

[~,~,L4] = modelClassification(trainData, trainLabels,'Quadratic')  
plot(L4,'o')
